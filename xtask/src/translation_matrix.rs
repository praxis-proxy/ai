// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask translation-matrix` validates the translation support matrix
//! manifest and regenerates (or verifies) `docs/conformance/translation-matrix.md`.
//!
//! The manifest is the published contract for what Praxis supports natively,
//! what it translates, and what it refuses. Validation is the point of this
//! task; rendering is a consequence. Two invariants carry the weight:
//!
//! - Every surface answers every feature in the shared catalog, so the table is a matrix rather than a pile of
//!   per-filter notes. A feature added to the catalog fails the check until each surface classifies it.
//! - A `dropped` cell is a defect, not a support level, and must carry a blocker. Those cells never reach the support
//!   table; they render into "Known gaps" so the document cannot claim behavior Praxis lacks.

use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask translation-matrix`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Write the generated document instead of checking for drift.
    #[arg(long)]
    fix: bool,
}

/// Matrix manifest, relative to the workspace root.
const MANIFEST_REL: &str = "docs/conformance/translation-matrix.yaml";

/// Generated document, relative to the workspace root.
const DOC_REL: &str = "docs/conformance/translation-matrix.md";

/// Manifest schema version this task understands.
const SUPPORTED_VERSION: u32 = 1;

// -----------------------------------------------------------------------------
// Manifest
// -----------------------------------------------------------------------------

/// How a surface handles one feature.
///
/// Seven values rather than three: `native` and `passthrough` are different
/// promises, a lossy translation needs somewhere to record what it loses, and
/// `dropped` needs to exist so the audit has a truthful place to put a field
/// that is accepted and then ignored.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Status {
    /// Praxis implements the operation locally and owns the payload.
    Native,
    /// Forwarded to the backend unchanged; backend semantics apply.
    Passthrough,
    /// Converted with no semantic loss.
    Translated,
    /// Converted, with something approximated.
    TranslatedLossy,
    /// Explicit dialect-shaped error at the proxy edge.
    Rejected,
    /// Accepted, then silently ignored or silently wrong. A defect.
    Dropped,
    /// The client API has no such concept.
    #[serde(rename = "n-a")]
    NotApplicable,
}

impl Status {
    /// Short label used in the rendered tables.
    const fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Passthrough => "passthrough",
            Self::Translated => "translated",
            Self::TranslatedLossy => "lossy",
            Self::Rejected => "rejected",
            Self::Dropped => "gap",
            Self::NotApplicable => "n/a",
        }
    }

    /// Whether this status records a defect rather than a support level.
    const fn is_gap(self) -> bool {
        matches!(self, Self::Dropped)
    }

    /// Whether a cell with this status must cite code.
    const fn needs_evidence(self) -> bool {
        !matches!(self, Self::NotApplicable)
    }

    /// Whether a cell with this status must explain itself.
    ///
    /// "Lossy" with no note is a claim with no content, and a gap with no note
    /// cannot be acted on.
    const fn needs_note(self) -> bool {
        matches!(self, Self::TranslatedLossy | Self::Dropped)
    }
}

/// A fact about pipeline composition that changes what every cell means.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Precondition {
    /// Stable identifier.
    id: String,
    /// One-paragraph statement of the precondition.
    summary: String,
    /// Code reference backing the claim.
    evidence: String,
    /// Supporting specifics, rendered as a nested list.
    detail: Vec<String>,
}

/// One row of the matrix, shared by every surface.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Feature {
    /// Stable identifier, used as the key in each surface's `cells` map.
    id: String,
    /// Human-readable row label.
    label: String,
}

/// An inbound surface Praxis serves without translating.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeSurface {
    /// Stable identifier.
    id: String,
    /// Dialect the client speaks.
    client_api: String,
    /// Operation count, where the surface has a bounded registry.
    operations: Option<u32>,
    /// How Praxis handles the surface.
    status: Status,
    /// Qualifications a reader needs.
    note: String,
    /// Code reference backing the claim.
    evidence: String,
}

/// One feature's treatment on one translation surface.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    /// How the surface handles the feature.
    status: Status,
    /// What is approximated, lost, or silently wrong.
    note: Option<String>,
    /// Code reference backing the claim.
    evidence: Option<String>,
    /// Tracking reference. Required for, and only for, a gap.
    blocker: Option<String>,
}

/// A client dialect translated onto a backend dialect.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Surface {
    /// Stable identifier.
    id: String,
    /// Column header in the matrix table.
    short: String,
    /// Dialect the client speaks.
    client_api: String,
    /// Dialect the backend speaks.
    backend_api: String,
    /// Filters implementing the translation.
    filters: Vec<String>,
    /// Filters that must also be installed for rejections to be dialect-shaped.
    clean_rejection_requires: Vec<String>,
    /// What the surface does with fields it does not recognize.
    unknown_field_policy: String,
    /// Missing or stale documentation for this surface, when known.
    doc_gap: Option<String>,
    /// Feature id to treatment. Must cover the catalog exactly.
    cells: std::collections::BTreeMap<String, Cell>,
}

impl Surface {
    /// Look up the cell for `feature`.
    ///
    /// Rendering runs only after [`validate`] has proven every surface
    /// classifies every catalog feature, so a missing key here is a bug in
    /// that check rather than a manifest problem.
    fn cell(&self, feature: &str) -> &Cell {
        self.cells
            .get(feature)
            .expect("validation guarantees every surface classifies every feature")
    }
}

/// A defect affecting more than one surface, recorded once.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossSurfaceGap {
    /// Stable identifier.
    id: String,
    /// Surfaces exhibiting the defect.
    affects: Vec<String>,
    /// What goes wrong, and why it matters to a customer.
    summary: String,
    /// Tracking reference.
    blocker: String,
}

/// The translation support matrix manifest.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Schema version.
    version: u32,
    /// Facts that qualify every cell below.
    preconditions: Vec<Precondition>,
    /// Shared row vocabulary.
    features: Vec<Feature>,
    /// Surfaces served without translation.
    native_surfaces: Vec<NativeSurface>,
    /// Qualification that must accompany the conformance coverage number.
    conformance_claim_caveat: String,
    /// Translation surfaces.
    surfaces: Vec<Surface>,
    /// Defects spanning surfaces.
    cross_surface_gaps: Vec<CrossSurfaceGap>,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Validate the manifest, then verify or regenerate the rendered document.
pub(crate) fn run(args: &Args) {
    let root = workspace_root();
    let manifest = load_manifest(&root.join(MANIFEST_REL));

    let problems = validate(&manifest);
    if !problems.is_empty() {
        eprintln!("{MANIFEST_REL} is invalid:");
        for problem in &problems {
            eprintln!("  {problem}");
        }
        std::process::exit(1);
    }

    emit(args, &root, &render_doc(&manifest));
}

/// Read and parse the manifest, reporting a schema mismatch and exiting.
fn load_manifest(path: &Path) -> Manifest {
    let raw = fs::read_to_string(path).expect("failed to read translation matrix manifest");
    match serde_yaml::from_str(&raw) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("{MANIFEST_REL} does not match the schema:\n  {err}");
            std::process::exit(1);
        },
    }
}

/// Write the rendered document, or verify the committed copy still matches.
fn emit(args: &Args, root: &Path, content: &str) {
    let doc_path = root.join(DOC_REL);

    if args.fix {
        fs::write(&doc_path, content).unwrap();
        println!("wrote {}", doc_path.strip_prefix(root).unwrap_or(&doc_path).display());
    } else if fs::read_to_string(&doc_path).unwrap_or_default() == content {
        println!("{DOC_REL} is up to date");
    } else {
        eprintln!("{DOC_REL} is stale");
        eprintln!("\nrun: cargo xtask translation-matrix --fix");
        std::process::exit(1);
    }
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Collect every schema problem, so one run reports all of them.
fn validate(m: &Manifest) -> Vec<String> {
    let mut problems = Vec::new();

    if m.version != SUPPORTED_VERSION {
        problems.push(format!(
            "version is {}, this task understands {SUPPORTED_VERSION}",
            m.version
        ));
    }

    let feature_ids = validate_features(m, &mut problems);
    let surface_ids = validate_surfaces(m, &feature_ids, &mut problems);
    validate_native_surfaces(m, &mut problems);
    validate_preconditions(m, &mut problems);
    validate_cross_surface_gaps(m, &surface_ids, &mut problems);

    problems
}

/// Check feature id uniqueness and return the catalog every surface answers.
fn validate_features<'a>(m: &'a Manifest, problems: &mut Vec<String>) -> BTreeSet<&'a str> {
    let mut ids = BTreeSet::new();
    for feature in &m.features {
        if !ids.insert(feature.id.as_str()) {
            problems.push(format!("duplicate feature id `{}`", feature.id));
        }
    }
    ids
}

/// Check surface id uniqueness and each surface's cells; return the ids.
fn validate_surfaces<'a>(
    m: &'a Manifest,
    feature_ids: &BTreeSet<&str>,
    problems: &mut Vec<String>,
) -> BTreeSet<&'a str> {
    let mut ids = BTreeSet::new();
    for surface in &m.surfaces {
        if !ids.insert(surface.id.as_str()) {
            problems.push(format!("duplicate surface id `{}`", surface.id));
        }
        validate_surface(surface, feature_ids, problems);
    }
    ids
}

/// Check native surface id uniqueness and that each carries a native status.
fn validate_native_surfaces(m: &Manifest, problems: &mut Vec<String>) {
    let mut ids = BTreeSet::new();
    for native in &m.native_surfaces {
        if !ids.insert(native.id.as_str()) {
            problems.push(format!("duplicate native surface id `{}`", native.id));
        }
        if !matches!(native.status, Status::Native | Status::Passthrough) {
            problems.push(format!(
                "native surface `{}` has status `{}`; expected native or passthrough",
                native.id,
                native.status.label()
            ));
        }
    }
}

/// Check that every precondition actually states something.
fn validate_preconditions(m: &Manifest, problems: &mut Vec<String>) {
    for precondition in &m.preconditions {
        if precondition.summary.trim().is_empty() {
            problems.push(format!("precondition `{}` has an empty summary", precondition.id));
        }
    }
}

/// Check that cross-surface gaps name real surfaces and carry a blocker.
fn validate_cross_surface_gaps(m: &Manifest, surface_ids: &BTreeSet<&str>, problems: &mut Vec<String>) {
    for gap in &m.cross_surface_gaps {
        if gap.affects.is_empty() {
            problems.push(format!("cross-surface gap `{}` affects nothing", gap.id));
        }
        for affected in &gap.affects {
            if !surface_ids.contains(affected.as_str()) {
                problems.push(format!(
                    "cross-surface gap `{}` references unknown surface `{affected}`",
                    gap.id
                ));
            }
        }
        if gap.blocker.trim().is_empty() {
            problems.push(format!("cross-surface gap `{}` has no blocker", gap.id));
        }
    }
}

/// Check one surface's cells against the feature catalog and the cell rules.
fn validate_surface(surface: &Surface, feature_ids: &BTreeSet<&str>, problems: &mut Vec<String>) {
    let present: BTreeSet<&str> = surface.cells.keys().map(String::as_str).collect();

    for missing in feature_ids.difference(&present) {
        problems.push(format!("surface `{}` does not classify `{missing}`", surface.id));
    }
    for unknown in present.difference(feature_ids) {
        problems.push(format!(
            "surface `{}` classifies `{unknown}`, which is not in the feature catalog",
            surface.id
        ));
    }

    for (feature, cell) in &surface.cells {
        let where_ = format!("surface `{}` feature `{feature}`", surface.id);

        match (&cell.blocker, cell.status.is_gap()) {
            (None, true) => problems.push(format!("{where_} is a gap with no blocker")),
            (Some(blocker), true) if blocker.trim().is_empty() => {
                problems.push(format!("{where_} is a gap with an empty blocker"));
            },
            (Some(_), false) => problems.push(format!(
                "{where_} has a blocker but status `{}`; only a gap takes a blocker",
                cell.status.label()
            )),
            _ => {},
        }

        if cell.status.needs_evidence() && cell.evidence.as_ref().is_none_or(|e| e.trim().is_empty()) {
            problems.push(format!("{where_} has status `{}` but no evidence", cell.status.label()));
        }

        if cell.status.needs_note() && cell.note.as_ref().is_none_or(|n| n.trim().is_empty()) {
            problems.push(format!("{where_} has status `{}` but no note", cell.status.label()));
        }
    }
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

/// Collapse a folded-scalar note onto one line so it fits a table cell.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Render the full `translation-matrix.md` document.
#[expect(clippy::too_many_lines, reason = "straight-line Markdown document template")]
fn render_doc(m: &Manifest) -> String {
    let mut out = String::new();

    writeln!(out, "# Translation Support Matrix").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "<!-- Generated by `cargo xtask translation-matrix --fix`. Do not edit"
    )
    .unwrap();
    writeln!(out, "     by hand — edit {MANIFEST_REL} and regenerate. -->").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "What Praxis AI serves natively, what it translates between dialects, and"
    )
    .unwrap();
    writeln!(out, "what it refuses.").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "## Reading the matrix").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Status | Meaning |").unwrap();
    writeln!(out, "| --- | --- |").unwrap();
    writeln!(
        out,
        "| `native` | Praxis implements the operation locally and owns the payload. |"
    )
    .unwrap();
    writeln!(
        out,
        "| `passthrough` | Forwarded to the backend unchanged; backend semantics apply. |"
    )
    .unwrap();
    writeln!(out, "| `translated` | Converted with no semantic loss. |").unwrap();
    writeln!(
        out,
        "| `lossy` | Converted, with something approximated. The surface section says what. |"
    )
    .unwrap();
    writeln!(out, "| `rejected` | Explicit dialect-shaped error at the proxy edge. |").unwrap();
    writeln!(
        out,
        "| `gap` | Known defect: accepted and then silently ignored or silently wrong. See [Known gaps](#known-gaps). |"
    )
    .unwrap();
    writeln!(out, "| `n/a` | The client API has no such concept. |").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "`gap` is not a support level. A gap cell means the request is accepted and"
    )
    .unwrap();
    writeln!(
        out,
        "the stated behavior does not happen — treat it as unsupported until the"
    )
    .unwrap();
    writeln!(out, "linked blocker closes.").unwrap();
    writeln!(out).unwrap();

    render_preconditions(m, &mut out);
    render_matrix(m, &mut out);
    render_gaps(m, &mut out);
    render_surfaces(m, &mut out);
    render_native(m, &mut out);

    out
}

/// Render the pipeline-composition facts that qualify every cell.
fn render_preconditions(m: &Manifest, out: &mut String) {
    writeln!(out, "## Before you read the matrix").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The cells below describe a pipeline that includes the filters named in each"
    )
    .unwrap();
    writeln!(
        out,
        "surface section. These facts change what a cell means in a given deployment."
    )
    .unwrap();
    writeln!(out).unwrap();

    for precondition in &m.preconditions {
        writeln!(out, "- **{}** — {}", precondition.id, one_line(&precondition.summary)).unwrap();
        for detail in &precondition.detail {
            writeln!(out, "  - {}", one_line(detail)).unwrap();
        }
        writeln!(out, "  - Evidence: `{}`", precondition.evidence).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the feature-by-surface table.
fn render_matrix(m: &Manifest, out: &mut String) {
    writeln!(out, "## Matrix").unwrap();
    writeln!(out).unwrap();

    write!(out, "| Feature |").unwrap();
    for surface in &m.surfaces {
        write!(out, " {} |", surface.short).unwrap();
    }
    writeln!(out).unwrap();

    write!(out, "| --- |").unwrap();
    for _ in &m.surfaces {
        write!(out, " --- |").unwrap();
    }
    writeln!(out).unwrap();

    for feature in &m.features {
        write!(out, "| {} |", feature.label).unwrap();
        for surface in &m.surfaces {
            let status = surface.cell(&feature.id).status;
            if status.is_gap() {
                write!(out, " [`gap`](#known-gaps) |").unwrap();
            } else {
                write!(out, " `{}` |", status.label()).unwrap();
            }
        }
        writeln!(out).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render every gap cell and every cross-surface defect.
fn render_gaps(m: &Manifest, out: &mut String) {
    writeln!(out, "## Known gaps").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Behavior Praxis does not provide despite accepting the request. These are"
    )
    .unwrap();
    writeln!(out, "defects, not support levels.").unwrap();
    writeln!(out).unwrap();

    render_cross_surface_gaps(m, out);
    render_per_surface_gaps(m, out);
}

/// Render defects recorded once because they span several surfaces.
fn render_cross_surface_gaps(m: &Manifest, out: &mut String) {
    writeln!(out, "### Spanning several surfaces").unwrap();
    writeln!(out).unwrap();
    for gap in &m.cross_surface_gaps {
        writeln!(out, "- **{}** — {}", gap.id, one_line(&gap.summary)).unwrap();
        writeln!(out, "  - Affects: {}", gap.affects.join(", ")).unwrap();
        writeln!(out, "  - Blocker: {}", gap.blocker).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render one row per gap cell, grouped by surface.
fn render_per_surface_gaps(m: &Manifest, out: &mut String) {
    writeln!(out, "### Per surface").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Surface | Feature | Gap | Blocker |").unwrap();
    writeln!(out, "| --- | --- | --- | --- |").unwrap();
    for surface in &m.surfaces {
        for feature in &m.features {
            let cell = surface.cell(&feature.id);
            if !cell.status.is_gap() {
                continue;
            }
            writeln!(
                out,
                "| {} | {} | {} | {} |",
                surface.short,
                feature.label,
                one_line(cell.note.as_deref().unwrap_or_default()),
                cell.blocker.as_deref().unwrap_or_default(),
            )
            .unwrap();
        }
    }
    writeln!(out).unwrap();
}

/// Render the per-surface detail sections.
fn render_surfaces(m: &Manifest, out: &mut String) {
    writeln!(out, "## Surfaces").unwrap();
    writeln!(out).unwrap();

    for surface in &m.surfaces {
        render_surface_header(surface, out);
        render_surface_table(m, surface, out);
    }
}

/// Render one surface's filters, rejection preconditions, and policies.
fn render_surface_header(surface: &Surface, out: &mut String) {
    writeln!(out, "### {} to {}", surface.client_api, surface.backend_api).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "- Filters: {}", surface.filters.join(", ")).unwrap();
    if surface.clean_rejection_requires.is_empty() {
        writeln!(out, "- Dialect-shaped rejection: emitted by the translator itself.").unwrap();
    } else {
        writeln!(
            out,
            "- Dialect-shaped rejection also requires: {}",
            surface.clean_rejection_requires.join(", ")
        )
        .unwrap();
    }
    writeln!(
        out,
        "- Unrecognized fields: {}",
        one_line(&surface.unknown_field_policy)
    )
    .unwrap();
    if let Some(doc_gap) = &surface.doc_gap {
        writeln!(out, "- Documentation gap: {}", one_line(doc_gap)).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render one surface's full feature table.
fn render_surface_table(m: &Manifest, surface: &Surface, out: &mut String) {
    writeln!(out, "| Feature | Status | Notes | Evidence |").unwrap();
    writeln!(out, "| --- | --- | --- | --- |").unwrap();
    for feature in &m.features {
        let cell = surface.cell(&feature.id);
        let evidence = cell.evidence.as_deref().map_or_else(String::new, |e| format!("`{e}`"));
        writeln!(
            out,
            "| {} | `{}` | {} | {} |",
            feature.label,
            cell.status.label(),
            one_line(cell.note.as_deref().unwrap_or_default()),
            evidence,
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the native-surface table and the conformance caveat.
fn render_native(m: &Manifest, out: &mut String) {
    writeln!(out, "## Native surfaces").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Surface | Operations | Status | Notes | Evidence |").unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for native in &m.native_surfaces {
        let operations = native.operations.map_or_else(|| "—".to_owned(), |n| n.to_string());
        writeln!(
            out,
            "| {} | {operations} | `{}` | {} | `{}` |",
            native.client_api,
            native.status.label(),
            one_line(&native.note),
            native.evidence,
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "{}", one_line(&m.conformance_claim_caveat)).unwrap();
}

// -----------------------------------------------------------------------------
// Paths
// -----------------------------------------------------------------------------

/// Resolve the workspace root from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate always has a parent directory")
        .to_path_buf()
}
