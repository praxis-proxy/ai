//! Coverage-manifest loading and inference fixture discovery.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use super::{
    FixtureError, InferenceProtocol, InferenceScenario, ProvenanceKind, WireFixture,
    schema::{LoadedWireFixture, PersistedDocumentLimits, load_persisted_document},
    validate_commit_safe,
};

/// The coverage state of a protocol-transformation feature.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum CoverageStatus {
    /// A replay test covers the feature.
    Covered,
    /// A replay test and a live-provider recording cover the feature.
    LiveCovered,
    /// A controlled synthetic backend is the coverage source.
    SyntheticOnly,
    /// Praxis deliberately does not support the feature.
    Unsupported,
    /// A named provider does not support the feature.
    ProviderUnsupported,
}

/// Coverage requirements for one feature in the transformation scope.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageFeature {
    /// Stable feature identifier.
    pub id: String,
    /// Transformation scopes whose denominator includes this feature.
    pub scopes: Vec<String>,
    /// The coverage state for this feature.
    pub status: CoverageStatus,
    /// Scenario identifiers that exercise the feature.
    pub scenarios: Vec<String>,
    /// Per-provider coverage requirements.
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderCoverage>,
    /// Required explanation for unsupported entries.
    pub reason: Option<String>,
}

/// Coverage requirements for a particular provider.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCoverage {
    /// The coverage state for this provider.
    pub status: CoverageStatus,
    /// Required explanation for unsupported entries.
    pub reason: Option<String>,
}

/// Versioned feature-coverage declaration for an inference fixture tree.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageManifest {
    /// The coverage-manifest schema version.
    pub version: u32,
    /// Stable transformation-scope identifiers.
    pub scope: Vec<String>,
    /// Features covered by the manifest.
    pub features: Vec<CoverageFeature>,
}

/// A recording discovered from its serialized fixture content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordingRef {
    /// Path of the recording fixture.
    pub path: PathBuf,
    /// Stable scenario identity declared by the fixture body.
    pub scenario_id: String,
    /// Inference protocol declared by the fixture body.
    pub protocol: InferenceProtocol,
    /// Provider identity declared by the fixture provenance.
    pub provider: String,
    /// The source category declared by the recording provenance.
    pub provenance_kind: ProvenanceKind,
}

/// One scenario document retained with the path from which it was loaded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScenarioSnapshot {
    /// Deterministically discovered scenario path.
    pub path: PathBuf,
    /// Strict scenario value loaded exactly once for this snapshot.
    pub scenario: InferenceScenario,
}

/// Aggregate coverage facts for a validated fixture tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoverageReport {
    /// Number of declared features.
    pub features_total: usize,
    /// Number of discovered scenarios.
    pub scenarios_total: usize,
    /// Number of discovered recordings.
    pub recordings_total: usize,
    /// Number of features in each coverage state.
    pub counts_by_status: BTreeMap<CoverageStatus, usize>,
}

/// One coherent recording discovery result with deferred commit-safety state.
struct RecordingDiscovery {
    /// Metadata retained for coverage-graph validation.
    recordings: Vec<RecordingRef>,
    /// First deterministic default-policy failure, if one was found.
    commit_safety_error: Option<FixtureError>,
}

impl CoverageManifest {
    /// Loads a coverage manifest from YAML.
    ///
    /// # Errors
    ///
    /// Returns an error when the bounded document cannot be read or allocated,
    /// strict YAML structure or typed parsing fails, or its schema version is
    /// unsupported.
    pub fn load(path: &Path) -> Result<Self, FixtureError> {
        Self::load_with_limits(path, PersistedDocumentLimits::COVERAGE)
    }

    /// Loads a manifest under explicit testable persisted-document ceilings.
    fn load_with_limits(path: &Path, limits: PersistedDocumentLimits) -> Result<Self, FixtureError> {
        let manifest: Self = load_persisted_document(path, limits)?.value;
        manifest.validate_version()?;
        Ok(manifest)
    }

    /// Verifies that this manifest uses coverage schema version 1.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::UnsupportedCoverageManifestVersion`] when the
    /// manifest version is not 1.
    pub fn validate_version(&self) -> Result<(), FixtureError> {
        if self.version == 1 {
            Ok(())
        } else {
            Err(FixtureError::UnsupportedCoverageManifestVersion { version: self.version })
        }
    }
}

/// Discovers YAML scenarios beneath `root/scenarios` by their declared IDs.
///
/// # Errors
///
/// Returns an error when discovery or scenario validation fails.
pub fn discover_scenarios(root: &Path) -> Result<BTreeMap<String, PathBuf>, FixtureError> {
    Ok(discover_scenario_snapshots(root)?
        .into_iter()
        .map(|(id, snapshot)| (id, snapshot.path))
        .collect())
}

/// Discovers and retains each strict-loaded scenario with its source path.
///
/// # Errors
///
/// Returns an error when discovery, scenario validation, or identity indexing
/// fails. Every returned scenario document was loaded exactly once.
pub fn discover_scenario_snapshots(root: &Path) -> Result<BTreeMap<String, ScenarioSnapshot>, FixtureError> {
    discover_scenario_snapshots_with_loader(root, InferenceScenario::load)
}

/// Scenario discovery with an injectable loader for snapshot/read-count tests.
fn discover_scenario_snapshots_with_loader(
    root: &Path,
    mut load_scenario: impl FnMut(&Path) -> Result<InferenceScenario, FixtureError>,
) -> Result<BTreeMap<String, ScenarioSnapshot>, FixtureError> {
    let paths = discover_files(&root.join("scenarios"), &["yaml", "yml"])?;
    let mut scenarios = BTreeMap::new();
    for path in paths {
        let scenario = load_scenario(&path)?;
        if scenario.id.trim().is_empty() {
            return invariant("scenario ID must not be empty");
        }
        let id = scenario.id.clone();
        if scenarios
            .insert(id.clone(), ScenarioSnapshot { path, scenario })
            .is_some()
        {
            return Err(FixtureError::DuplicateScenarioId { id });
        }
    }
    Ok(scenarios)
}

/// Discovers JSON recordings beneath `root/recordings` from fixture content.
///
/// # Errors
///
/// Returns an error when discovery or fixture validation fails.
pub fn discover_recordings(root: &Path) -> Result<Vec<RecordingRef>, FixtureError> {
    let discovery = discover_recordings_with_fixture_loader(root, WireFixture::load_for_commit_check)?;
    if let Some(error) = discovery.commit_safety_error {
        return Err(error);
    }
    Ok(discovery.recordings)
}

/// Strictly loads recordings once while retaining only coverage metadata.
fn discover_recordings_with_fixture_loader(
    root: &Path,
    mut load_fixture: impl FnMut(&Path) -> Result<LoadedWireFixture, FixtureError>,
) -> Result<RecordingDiscovery, FixtureError> {
    let paths = discover_files(&root.join("recordings"), &["json"])?;
    let mut recordings = Vec::with_capacity(paths.len());
    let mut identities = BTreeMap::new();
    let mut commit_safety_error = None;
    for path in paths {
        let loaded = load_fixture(&path)?;
        recordings.push(recording_ref(path, loaded, &mut identities, &mut commit_safety_error)?);
    }
    Ok(RecordingDiscovery {
        recordings,
        commit_safety_error,
    })
}

/// Validates one loaded fixture and moves its coverage metadata into a reference.
fn recording_ref(
    path: PathBuf,
    loaded: LoadedWireFixture,
    identities: &mut BTreeMap<(String, String), PathBuf>,
    commit_safety_error: &mut Option<FixtureError>,
) -> Result<RecordingRef, FixtureError> {
    let fixture = loaded.fixture;
    if commit_safety_error.is_none() {
        *commit_safety_error = validate_commit_safe(&fixture).err().or(loaded.raw_commit_safety_error);
    }
    let identity = (fixture.scenario_id.clone(), fixture.provenance.provider.clone());
    if let Some(first_path) = identities.insert(identity.clone(), path.clone()) {
        return Err(FixtureError::DuplicateRecordingIdentity {
            scenario_id: identity.0,
            provider: identity.1,
            first_path,
            second_path: path,
        });
    }
    Ok(RecordingRef {
        path,
        scenario_id: fixture.scenario_id,
        protocol: fixture.protocol,
        provider: fixture.provenance.provider,
        provenance_kind: fixture.provenance.kind,
    })
}

/// Validates the manifest, scenario graph, and provider recording requirements.
///
/// # Errors
///
/// Returns an error when any coverage invariant is not satisfied.
pub fn check_coverage(root: &Path) -> Result<CoverageReport, FixtureError> {
    check_coverage_with_fixture_loader(root, WireFixture::load_for_commit_check)
}

/// Coverage implementation with an injectable recording loader for read-count tests.
fn check_coverage_with_fixture_loader(
    root: &Path,
    load_fixture: impl FnMut(&Path) -> Result<LoadedWireFixture, FixtureError>,
) -> Result<CoverageReport, FixtureError> {
    check_coverage_with_loaders(root, InferenceScenario::load, load_fixture)
}

/// Coverage implementation with injectable coherent document loaders.
fn check_coverage_with_loaders(
    root: &Path,
    load_scenario: impl FnMut(&Path) -> Result<InferenceScenario, FixtureError>,
    load_fixture: impl FnMut(&Path) -> Result<LoadedWireFixture, FixtureError>,
) -> Result<CoverageReport, FixtureError> {
    let manifest = CoverageManifest::load(&root.join("coverage.yaml"))?;
    validate_manifest_shape(&manifest)?;
    let scenarios = discover_scenario_snapshots_with_loader(root, load_scenario)?;
    let features = index_features(&manifest.features)?;

    validate_scenario_feature_links(&scenarios, &features)?;
    validate_feature_scenario_links(&manifest.features, &scenarios)?;
    let discovery = discover_recordings_with_fixture_loader(root, load_fixture)?;
    validate_recording_scenarios(&discovery.recordings, &scenarios)?;
    validate_feature_recordings(&manifest.features, &discovery.recordings)?;
    if let Some(error) = discovery.commit_safety_error {
        return Err(error);
    }

    Ok(coverage_report(
        &manifest.features,
        scenarios.len(),
        discovery.recordings.len(),
    ))
}

/// Builds a stable lookup table after validating feature metadata and identity.
fn index_features(features: &[CoverageFeature]) -> Result<BTreeMap<&str, &CoverageFeature>, FixtureError> {
    let mut index = BTreeMap::new();
    for feature in features {
        if feature.id.trim().is_empty() {
            return invariant("feature ID must not be empty");
        }
        if feature.scenarios.is_empty() {
            return invariant(&format!(
                "feature `{}` must reference at least one scenario",
                feature.id
            ));
        }
        let mut scenario_ids = BTreeSet::new();
        for scenario_id in &feature.scenarios {
            if scenario_id.trim().is_empty() {
                return invariant(&format!("feature `{}` has an empty scenario ID", feature.id));
            }
            if !scenario_ids.insert(scenario_id) {
                return invariant(&format!("feature `{}` repeats scenario `{scenario_id}`", feature.id));
            }
        }
        if index.insert(feature.id.as_str(), feature).is_some() {
            return Err(FixtureError::DuplicateCoverageFeatureId { id: feature.id.clone() });
        }
        validate_feature_metadata(feature)?;
    }
    Ok(index)
}

/// Verifies links declared from each scenario to a manifest feature.
fn validate_scenario_feature_links(
    scenarios: &BTreeMap<String, ScenarioSnapshot>,
    features: &BTreeMap<&str, &CoverageFeature>,
) -> Result<(), FixtureError> {
    for (scenario_id, snapshot) in scenarios {
        let scenario = &snapshot.scenario;
        if scenario.features.is_empty() {
            return Err(FixtureError::CoverageScenarioWithoutFeatures {
                scenario_id: scenario_id.clone(),
            });
        }
        let mut scenario_features = BTreeSet::new();
        for feature_id in &scenario.features {
            if feature_id.trim().is_empty() {
                return invariant(&format!("scenario `{scenario_id}` has an empty feature ID"));
            }
            if !scenario_features.insert(feature_id) {
                return invariant(&format!("scenario `{scenario_id}` repeats feature `{feature_id}`"));
            }
            let feature = features
                .get(feature_id.as_str())
                .ok_or_else(|| FixtureError::CoverageUnknownFeature {
                    scenario_id: scenario_id.clone(),
                    feature_id: feature_id.clone(),
                })?;
            if !feature.scenarios.iter().any(|id| id == scenario_id) {
                return Err(FixtureError::CoverageOneSidedLink {
                    feature_id: feature.id.clone(),
                    scenario_id: scenario_id.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Verifies links declared from each manifest feature to a scenario.
fn validate_feature_scenario_links(
    features: &[CoverageFeature],
    scenarios: &BTreeMap<String, ScenarioSnapshot>,
) -> Result<(), FixtureError> {
    for feature in features {
        for scenario_id in &feature.scenarios {
            let snapshot = scenarios
                .get(scenario_id)
                .ok_or_else(|| FixtureError::CoverageMissingScenario {
                    feature_id: feature.id.clone(),
                    scenario_id: scenario_id.clone(),
                })?;
            let scenario = &snapshot.scenario;
            if !scenario.features.iter().any(|id| id == &feature.id) {
                return Err(FixtureError::CoverageOneSidedLink {
                    feature_id: feature.id.clone(),
                    scenario_id: scenario_id.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Verifies that every recording declares a discovered scenario.
fn validate_recording_scenarios(
    recordings: &[RecordingRef],
    scenarios: &BTreeMap<String, ScenarioSnapshot>,
) -> Result<(), FixtureError> {
    for recording in recordings {
        let snapshot =
            scenarios
                .get(&recording.scenario_id)
                .ok_or_else(|| FixtureError::CoverageUnknownRecordingScenario {
                    path: recording.path.clone(),
                    scenario_id: recording.scenario_id.clone(),
                })?;
        if recording.protocol != snapshot.scenario.protocol {
            return Err(FixtureError::CoverageRecordingProtocolMismatch {
                path: recording.path.clone(),
                scenario_id: recording.scenario_id.clone(),
            });
        }
    }
    Ok(())
}

/// Verifies that recording provenance satisfies feature-status requirements.
fn validate_feature_recordings(features: &[CoverageFeature], recordings: &[RecordingRef]) -> Result<(), FixtureError> {
    for feature in features {
        validate_recording_requirements(feature, recordings)?;
    }
    Ok(())
}

/// Builds the aggregate count fields exposed after successful validation.
fn coverage_report(features: &[CoverageFeature], scenarios_total: usize, recordings_total: usize) -> CoverageReport {
    let mut counts_by_status = BTreeMap::new();
    for feature in features {
        *counts_by_status.entry(feature.status.clone()).or_insert(0) += 1;
    }
    CoverageReport {
        features_total: features.len(),
        scenarios_total,
        recordings_total,
        counts_by_status,
    }
}

/// Recursively collects files with one of the permitted lower-case extensions.
///
/// Paths are sorted before traversal and symlinks are rejected, so fixture
/// discovery is deterministic and cannot escape the requested fixture tree.
fn discover_files(directory: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>, FixtureError> {
    let metadata = fs::symlink_metadata(directory).map_err(|source| FixtureError::ReadDirectory {
        path: directory.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(FixtureError::FixtureSymlink {
            path: directory.to_path_buf(),
        });
    }
    let mut paths = Vec::new();
    collect_files(directory, extensions, &mut paths)?;
    paths.sort();
    Ok(paths)
}

/// Adds eligible descendants of `directory` to `paths` in deterministic order.
fn collect_files(directory: &Path, extensions: &[&str], paths: &mut Vec<PathBuf>) -> Result<(), FixtureError> {
    let mut entries = fs::read_dir(directory)
        .map_err(|source| FixtureError::ReadDirectory {
            path: directory.to_path_buf(),
            source,
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| FixtureError::ReadDirectory {
            path: directory.to_path_buf(),
            source,
        })?;
    entries.sort_by_key(fs::DirEntry::path);
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| FixtureError::ReadDirectory {
            path: path.clone(),
            source,
        })?;
        if file_type.is_symlink() {
            return Err(FixtureError::FixtureSymlink { path });
        }
        if file_type.is_dir() {
            collect_files(&path, extensions, paths)?;
        } else if file_type.is_file() && has_extension(&path, extensions) {
            paths.push(path);
        }
    }
    Ok(())
}

/// Returns whether `path` has an extension accepted by this discovery pass.
fn has_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extensions.contains(&extension))
}

/// Checks unsupported-state explanations and provider declarations.
fn validate_feature_metadata(feature: &CoverageFeature) -> Result<(), FixtureError> {
    validate_feature_reason(feature)?;
    validate_provider_unsupported_declaration(feature)?;
    validate_provider_entries(feature)
}

/// Checks whether the feature-level status requires a reviewed reason.
fn validate_feature_reason(feature: &CoverageFeature) -> Result<(), FixtureError> {
    if requires_reason(&feature.status) && !has_reason(feature.reason.as_deref()) {
        return Err(FixtureError::CoverageReasonRequired {
            feature_id: feature.id.clone(),
            status: status_name(&feature.status).to_owned(),
        });
    }
    Ok(())
}

/// Requires provider-unsupported features to name an affected provider.
fn validate_provider_unsupported_declaration(feature: &CoverageFeature) -> Result<(), FixtureError> {
    if feature.status == CoverageStatus::ProviderUnsupported
        && (feature.providers.is_empty()
            || !feature
                .providers
                .values()
                .any(|coverage| coverage.status == CoverageStatus::ProviderUnsupported))
    {
        return invariant(&format!(
            "feature `{}` must name a provider_unsupported provider",
            feature.id
        ));
    }
    Ok(())
}

/// Checks provider names and status-specific provider reasons.
fn validate_provider_entries(feature: &CoverageFeature) -> Result<(), FixtureError> {
    for (provider, coverage) in &feature.providers {
        if provider.trim().is_empty() {
            return invariant(&format!("feature `{}` has an empty provider name", feature.id));
        }
        if requires_reason(&coverage.status) && !has_reason(coverage.reason.as_deref()) {
            return Err(FixtureError::CoverageProviderReasonRequired {
                feature_id: feature.id.clone(),
                provider: provider.clone(),
                status: status_name(&coverage.status).to_owned(),
            });
        }
    }
    Ok(())
}

/// Checks whether recordings provide the evidence required by `feature`.
fn validate_recording_requirements(feature: &CoverageFeature, recordings: &[RecordingRef]) -> Result<(), FixtureError> {
    for scenario_id in &feature.scenarios {
        let evidence = recordings
            .iter()
            .filter(|recording| recording.scenario_id == *scenario_id)
            .collect::<Vec<_>>();
        validate_top_level_evidence(feature, scenario_id, &evidence)?;
        validate_provider_evidence(feature, scenario_id, &evidence)?;
    }
    Ok(())
}

/// Validates non-provider-specific evidence required for one feature scenario.
fn validate_top_level_evidence(
    feature: &CoverageFeature,
    scenario_id: &str,
    evidence: &[&RecordingRef],
) -> Result<(), FixtureError> {
    let synthetic = |recording: &&RecordingRef| recording.provenance_kind == ProvenanceKind::Synthetic;
    let nonsynthetic = |recording: &&RecordingRef| recording.provenance_kind != ProvenanceKind::Synthetic;
    let valid = match feature.status {
        CoverageStatus::Covered => evidence
            .iter()
            .any(|recording| nonsynthetic(recording) && non_synthetic_provider_is_allowed(feature, recording)),
        CoverageStatus::LiveCovered => evidence.iter().any(|recording| {
            recording.provenance_kind == ProvenanceKind::Live
                && feature
                    .providers
                    .get(&recording.provider)
                    .is_some_and(|provider| is_supported(&provider.status))
        }),
        CoverageStatus::SyntheticOnly | CoverageStatus::Unsupported => evidence.iter().any(synthetic),
        CoverageStatus::ProviderUnsupported => evidence.iter().any(|recording| {
            synthetic(recording) || (nonsynthetic(recording) && non_synthetic_provider_is_allowed(feature, recording))
        }),
    };
    if valid {
        Ok(())
    } else {
        invariant(&format!(
            "feature `{}` lacks required {} evidence for scenario `{scenario_id}`",
            feature.id,
            status_name(&feature.status)
        ))
    }
}

/// Validates every provider declaration against evidence for one scenario.
fn validate_provider_evidence(
    feature: &CoverageFeature,
    scenario_id: &str,
    evidence: &[&RecordingRef],
) -> Result<(), FixtureError> {
    for (provider, coverage) in &feature.providers {
        let provider_evidence = evidence
            .iter()
            .filter(|recording| recording.provider == *provider)
            .collect::<Vec<_>>();
        let valid = match coverage.status {
            CoverageStatus::Covered => provider_evidence
                .iter()
                .any(|recording| recording.provenance_kind != ProvenanceKind::Synthetic),
            CoverageStatus::LiveCovered => provider_evidence
                .iter()
                .any(|recording| recording.provenance_kind == ProvenanceKind::Live),
            CoverageStatus::SyntheticOnly => provider_evidence
                .iter()
                .any(|recording| recording.provenance_kind == ProvenanceKind::Synthetic),
            CoverageStatus::Unsupported | CoverageStatus::ProviderUnsupported => !provider_evidence
                .iter()
                .any(|recording| recording.provenance_kind != ProvenanceKind::Synthetic),
        };
        if !valid {
            return invariant(&format!(
                "feature `{}` provider `{provider}` lacks valid {} evidence for scenario `{scenario_id}`",
                feature.id,
                status_name(&coverage.status)
            ));
        }
    }
    Ok(())
}

/// Validates top-level manifest denominator fields before discovery.
fn validate_manifest_shape(manifest: &CoverageManifest) -> Result<(), FixtureError> {
    if manifest.scope.is_empty() || manifest.features.is_empty() {
        return invariant("scope and features must not be empty");
    }
    let mut scope = BTreeSet::new();
    for id in &manifest.scope {
        if id.trim().is_empty() || !scope.insert(id.as_str()) {
            return invariant("scope IDs must be nonempty and unique");
        }
    }
    validate_feature_scope_links(&manifest.features, &scope)
}

/// Validates bidirectional membership between declared scopes and features.
fn validate_feature_scope_links(
    features: &[CoverageFeature],
    declared_scopes: &BTreeSet<&str>,
) -> Result<(), FixtureError> {
    let mut used_scopes = BTreeSet::new();
    for feature in features {
        if feature.scopes.is_empty() {
            return invariant(&format!("feature `{}` must reference at least one scope", feature.id));
        }
        let mut feature_scopes = BTreeSet::new();
        for scope_id in &feature.scopes {
            if scope_id.trim().is_empty() {
                return invariant(&format!("feature `{}` has an empty scope ID", feature.id));
            }
            if !feature_scopes.insert(scope_id.as_str()) {
                return invariant(&format!("feature `{}` repeats scope `{scope_id}`", feature.id));
            }
            if !declared_scopes.contains(scope_id.as_str()) {
                return invariant(&format!(
                    "feature `{}` references undeclared scope `{scope_id}`",
                    feature.id
                ));
            }
            used_scopes.insert(scope_id.as_str());
        }
    }
    if let Some(unused_scope) = declared_scopes.difference(&used_scopes).next() {
        return invariant(&format!("scope `{unused_scope}` is not referenced by any feature"));
    }
    Ok(())
}

/// Returns whether a provider status may satisfy non-synthetic evidence.
fn is_supported(status: &CoverageStatus) -> bool {
    matches!(status, CoverageStatus::Covered | CoverageStatus::LiveCovered)
}

/// Returns whether a non-synthetic recording obeys an optional provider policy.
fn non_synthetic_provider_is_allowed(feature: &CoverageFeature, recording: &RecordingRef) -> bool {
    (feature.status != CoverageStatus::ProviderUnsupported && feature.providers.is_empty())
        || feature
            .providers
            .get(&recording.provider)
            .is_some_and(|provider| is_supported(&provider.status))
}

/// Constructs a typed coverage-invariant failure.
fn invariant<T>(message: &str) -> Result<T, FixtureError> {
    Err(FixtureError::CoverageInvariant {
        message: message.to_owned(),
    })
}

/// Returns whether a status requires a human-reviewed explanation.
fn requires_reason(status: &CoverageStatus) -> bool {
    matches!(
        status,
        CoverageStatus::Unsupported | CoverageStatus::ProviderUnsupported
    )
}

/// Returns whether a reason contains non-whitespace text.
fn has_reason(reason: Option<&str>) -> bool {
    reason.is_some_and(|reason| !reason.trim().is_empty())
}

/// Returns the serialized spelling of a coverage status for diagnostics.
fn status_name(status: &CoverageStatus) -> &'static str {
    match status {
        CoverageStatus::Covered => "covered",
        CoverageStatus::LiveCovered => "live_covered",
        CoverageStatus::SyntheticOnly => "synthetic_only",
        CoverageStatus::Unsupported => "unsupported",
        CoverageStatus::ProviderUnsupported => "provider_unsupported",
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use serde_json::json;
    use tempfile::TempDir;

    use super::{
        CoverageManifest, CoverageStatus, check_coverage, check_coverage_with_fixture_loader,
        check_coverage_with_loaders, discover_recordings, discover_scenarios,
    };
    use crate::inference_fixture::{
        BodyKind, FixtureError, InferenceProtocol, InferenceScenario, RecordedBody, WireFixture,
        schema::{DocumentValidationLimits, PersistedDocumentKind, PersistedDocumentLimits},
    };

    #[test]
    fn accepts_valid_bidirectional_feature_and_scenario_links() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "one.json", "case/one", "source_a", "imported");

        let report = check_coverage(root.path()).unwrap();

        assert_eq!(report.features_total, 1);
        assert_eq!(report.scenarios_total, 1);
        assert_eq!(report.recordings_total, 1);
        assert_eq!(report.counts_by_status.get(&CoverageStatus::Covered), Some(&1));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "both vacuous coverage document categories are constructed independently"
    )]
    fn coverage_rejects_vacuous_scenario_and_recording_evidence() {
        let recording_root = fixture_root();
        write_manifest(recording_root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(recording_root.path(), "case/one", &["feature.one"]);
        write_recording(recording_root.path(), "one.json", "case/one", "source_a", "imported");
        let recording_path = recording_root.path().join("recordings/one.json");
        let mut recording: serde_json::Value = serde_json::from_slice(&fs::read(&recording_path).unwrap()).unwrap();
        recording["turns"] = json!([]);
        fs::write(&recording_path, serde_json::to_vec_pretty(&recording).unwrap()).unwrap();

        let recording_error = check_coverage(recording_root.path()).unwrap_err();

        let scenario_root = fixture_root();
        write_manifest(scenario_root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(scenario_root.path(), "case/one", &["feature.one"]);
        write_recording(scenario_root.path(), "one.json", "case/one", "source_a", "imported");
        let scenario_path = scenario_root.path().join("scenarios/scenario.yaml");
        let mut scenario: InferenceScenario = serde_yaml::from_slice(&fs::read(&scenario_path).unwrap()).unwrap();
        scenario.turns.clear();
        fs::write(&scenario_path, serde_yaml::to_string(&scenario).unwrap()).unwrap();

        let scenario_error = check_coverage(scenario_root.path()).unwrap_err();

        assert!(matches!(
            recording_error,
            FixtureError::InvalidInferenceTurnCount {
                document: "wire fixture",
                count: 0
            }
        ));
        assert!(matches!(
            scenario_error,
            FixtureError::InvalidInferenceTurnCount {
                document: "inference scenario",
                count: 0
            }
        ));
    }

    #[test]
    fn coverage_check_loads_each_recording_document_once() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "one.json", "case/one", "source_a", "imported");
        let mut reads = std::collections::BTreeMap::new();

        let report = check_coverage_with_fixture_loader(root.path(), |path| {
            *reads.entry(path.to_path_buf()).or_insert(0_usize) += 1;
            WireFixture::load_for_commit_check(path)
        })
        .unwrap();

        assert_eq!(report.recordings_total, 1);
        assert_eq!(reads.len(), 1);
        assert!(reads.values().all(|count| *count == 1));
    }

    #[test]
    fn coverage_check_loads_each_scenario_document_once() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "one.json", "case/one", "source_a", "imported");
        let mut reads = std::collections::BTreeMap::new();

        let report = check_coverage_with_loaders(
            root.path(),
            |path| {
                *reads.entry(path.to_path_buf()).or_insert(0_usize) += 1;
                InferenceScenario::load(path)
            },
            WireFixture::load_for_commit_check,
        )
        .unwrap();

        assert_eq!(report.scenarios_total, 1);
        assert_eq!(reads.len(), 1);
        assert!(reads.values().all(|count| *count == 1));
    }

    #[test]
    fn coverage_check_uses_one_scenario_snapshot_across_both_graph_directions() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "one.json", "case/one", "source_a", "imported");
        let mut mutated = false;

        let report = check_coverage_with_loaders(
            root.path(),
            |path| {
                let snapshot = InferenceScenario::load(path)?;
                write_scenario(root.path(), "case/one", &["feature.changed"]);
                mutated = true;
                Ok(snapshot)
            },
            WireFixture::load_for_commit_check,
        )
        .expect("both link directions should consume the retained pre-mutation scenario");

        assert!(mutated);
        assert_eq!(report.scenarios_total, 1);
        let changed = InferenceScenario::load(&root.path().join("scenarios/scenario.yaml")).unwrap();
        assert_eq!(changed.features, ["feature.changed"]);
    }

    #[test]
    fn rejects_duplicate_declared_scenario_ids() {
        let root = fixture_root();
        write_scenario_at(root.path(), "a/first.yaml", "case/one", &["feature.one"]);
        write_scenario_at(root.path(), "z/second.yml", "case/one", &["feature.one"]);

        let error = discover_scenarios(root.path()).unwrap_err();

        assert!(error.to_string().contains("duplicate scenario ID `case/one`"));
    }

    #[test]
    fn rejects_scenario_links_to_unknown_features() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.unknown"]);

        let error = check_coverage(root.path()).unwrap_err();

        assert!(error.to_string().contains("unknown feature `feature.unknown`"));
    }

    #[test]
    fn rejects_feature_links_to_missing_scenarios() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/missing", "");

        let error = check_coverage(root.path()).unwrap_err();

        assert!(error.to_string().contains("missing scenario `case/missing`"));
    }

    #[test]
    fn rejects_empty_reasons_for_unsupported_entries() {
        let root = fixture_root();
        write_manifest(root.path(), "unsupported", "feature.one", "case/one", "reason: '   '");
        write_scenario(root.path(), "case/one", &["feature.one"]);

        let error = check_coverage(root.path()).unwrap_err();

        assert!(error.to_string().contains("requires a nonempty reason"));
    }

    #[test]
    fn rejects_empty_reasons_for_provider_unsupported_entries() {
        let root = fixture_root();
        write_manifest(
            root.path(),
            "covered",
            "feature.one",
            "case/one",
            "providers:\n  source_a:\n    status: provider_unsupported\n    reason: ''",
        );
        write_scenario(root.path(), "case/one", &["feature.one"]);

        let error = check_coverage(root.path()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("provider `source_a` requires a nonempty reason")
        );
    }

    #[test]
    fn rejects_live_covered_features_without_a_matching_live_provider_recording() {
        let root = fixture_root();
        write_manifest(
            root.path(),
            "live_covered",
            "feature.one",
            "case/one",
            "providers:\n  source_a:\n    status: covered",
        );
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "other.json", "case/one", "source_b", "live");

        let error = check_coverage(root.path()).unwrap_err();

        assert!(error.to_string().contains("lacks required live_covered evidence"));
    }

    #[test]
    fn rejects_recordings_that_declare_an_unknown_scenario_id() {
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "case-one.json", "case/missing", "source_a", "synthetic");

        let error = check_coverage(root.path()).unwrap_err();

        assert!(error.to_string().contains("declares unknown scenario `case/missing`"));
    }

    #[test]
    fn rejects_recordings_with_a_mismatched_scenario_protocol() {
        // Catches discarding recording protocol metadata before coverage validation.
        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "case-one.json", "case/one", "source_a", "imported");
        let recording_path = root.path().join("recordings/case-one.json");
        let mut recording: serde_json::Value = serde_json::from_slice(&fs::read(&recording_path).unwrap()).unwrap();
        recording["protocol"] = json!("openai_responses");
        fs::write(&recording_path, serde_json::to_vec_pretty(&recording).unwrap()).unwrap();

        let error = check_coverage(root.path()).unwrap_err();

        assert!(matches!(
            &error,
            FixtureError::CoverageRecordingProtocolMismatch { path, scenario_id }
                if path == &recording_path && scenario_id == "case/one"
        ));
        let rendered = error.to_string();
        assert!(!rendered.contains("anthropic_messages"));
        assert!(!rendered.contains("openai_responses"));
    }

    #[test]
    fn accepts_a_synthetic_only_feature_with_a_synthetic_recording() {
        let root = fixture_root();
        write_manifest(root.path(), "synthetic_only", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(
            root.path(),
            "any-name.json",
            "case/one",
            "controlled_backend",
            "synthetic",
        );

        let report = check_coverage(root.path()).unwrap();

        assert_eq!(report.recordings_total, 1);
        assert_eq!(report.counts_by_status.get(&CoverageStatus::SyntheticOnly), Some(&1));
    }

    #[test]
    fn discovers_scenarios_and_recordings_in_deterministic_order() {
        let root = fixture_root();
        write_scenario_at(root.path(), "z/late.yaml", "case/z", &["feature.z"]);
        write_scenario_at(root.path(), "a/early.yml", "case/a", &["feature.a"]);
        write_scenario_at(root.path(), "ignored.txt", "case/ignored", &["feature.ignored"]);
        write_recording(root.path(), "z/late.json", "case/z", "source_z", "synthetic");
        write_recording(root.path(), "a/early.json", "case/a", "source_a", "synthetic");
        fs::write(root.path().join("recordings/ignored.yaml"), "not a recording").unwrap();

        let scenarios = discover_scenarios(root.path()).unwrap();
        let recordings = discover_recordings(root.path()).unwrap();

        assert_eq!(scenarios.keys().collect::<Vec<_>>(), vec!["case/a", "case/z"]);
        assert_eq!(
            recordings
                .iter()
                .map(|recording| recording.scenario_id.as_str())
                .collect::<Vec<_>>(),
            vec!["case/a", "case/z"]
        );
    }

    #[test]
    fn rejects_unknown_manifest_fields_and_unsupported_versions() {
        let root = fixture_root();
        fs::write(
            root.path().join("coverage.yaml"),
            "version: 1\nscope: []\nfeatures: []\nextra: rejected\n",
        )
        .unwrap();

        let error = CoverageManifest::load(&root.path().join("coverage.yaml")).unwrap_err();

        assert!(error.to_string().contains("failed to parse YAML fixture"));

        fs::write(
            root.path().join("coverage.yaml"),
            "version: 2\nscope: []\nfeatures: []\n",
        )
        .unwrap();

        let error = CoverageManifest::load(&root.path().join("coverage.yaml")).unwrap_err();

        assert!(error.to_string().contains("unsupported coverage manifest version 2"));

        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one]\n    providers:\n      source_a:\n        status: covered\n        unexpected: rejected\n",
        );
        let error = CoverageManifest::load(&root.path().join("coverage.yaml")).unwrap_err();
        assert!(error.to_string().contains("failed to parse YAML fixture"));
    }

    #[test]
    fn coverage_manifest_loader_bounds_bytes_and_structure_before_materialization() {
        let root = fixture_root();
        let path = root.path().join("coverage.yaml");
        fs::write(&path, "version: 1\nscope: [messages]\nfeatures: []\n").unwrap();

        let byte_error = CoverageManifest::load_with_limits(
            &path,
            PersistedDocumentLimits {
                max_encoded_bytes: 8,
                validation: DocumentValidationLimits::TEST_PERMISSIVE,
                kind: PersistedDocumentKind::Coverage,
            },
        )
        .unwrap_err();
        assert!(matches!(byte_error, FixtureError::PersistedDocumentTooLarge { .. }));

        let node_error = CoverageManifest::load_with_limits(
            &path,
            PersistedDocumentLimits {
                max_encoded_bytes: 1_000,
                validation: DocumentValidationLimits {
                    max_nodes: 2,
                    ..DocumentValidationLimits::TEST_PERMISSIVE
                },
                kind: PersistedDocumentKind::Coverage,
            },
        )
        .unwrap_err();
        assert!(node_error.to_string().contains("failed to parse YAML fixture"));
    }

    #[test]
    fn coverage_public_loader_rejects_its_dedicated_ceiling_before_allocation() {
        let root = fixture_root();
        let path = root.path().join("coverage.yaml");
        fs::File::create(&path)
            .unwrap()
            .set_len(u64::try_from(super::super::schema::MAX_COVERAGE_DOCUMENT_BYTES).unwrap() + 1)
            .unwrap();

        assert!(matches!(
            CoverageManifest::load(&path),
            Err(FixtureError::PersistedDocumentTooLarge {
                kind: "coverage manifest"
            })
        ));
    }

    #[test]
    fn coverage_manifest_rejects_duplicate_keys_and_multiple_yaml_documents() {
        let root = fixture_root();
        for document in [
            "version: 1\nversion: 1\nscope: []\nfeatures: []\n",
            "version: 1\nscope: []\nfeatures: []\n---\nversion: 1\nscope: []\nfeatures: []\n",
        ] {
            write_manifest_document(root.path(), document);
            assert!(CoverageManifest::load(&root.path().join("coverage.yaml")).is_err());
        }
    }

    #[test]
    fn coverage_manifest_loader_matches_typed_yaml_for_a_local_tagged_status() {
        let root = fixture_root();
        let document = "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: !live_covered\n    scenarios: [case/one]\n";
        let expected: CoverageManifest = serde_yaml::from_str(document).unwrap();
        write_manifest_document(root.path(), document);

        let loaded = CoverageManifest::load(&root.path().join("coverage.yaml")).unwrap();

        assert_eq!(loaded.features[0].status, expected.features[0].status);
        assert_eq!(loaded.features[0].status, CoverageStatus::LiveCovered);
    }

    #[test]
    fn coverage_public_loader_matches_typed_yaml_for_local_tagged_mapping_keys() {
        let root = fixture_root();
        let document = "!schema version: 1\n!coverage scope: [messages]\n!inventory features:\n  - !identity id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one]\n";
        let expected: CoverageManifest = serde_yaml::from_str(document).unwrap();
        write_manifest_document(root.path(), document);

        let loaded = CoverageManifest::load(&root.path().join("coverage.yaml")).unwrap();

        assert_eq!(loaded.version, expected.version);
        assert_eq!(loaded.scope, expected.scope);
        assert_eq!(loaded.features[0].id, expected.features[0].id);
        assert_eq!(loaded.features[0].status, expected.features[0].status);
    }

    #[test]
    fn coverage_loader_applies_small_budgets_to_local_tagged_mapping_keys() {
        let root = fixture_root();
        let path = root.path().join("coverage.yaml");
        let ordinary = "version: 1\nscope: [messages]\nfeatures: []\n";
        let document = "!schema version: 1\nscope: [messages]\nfeatures: []\n";
        let expected: CoverageManifest = serde_yaml::from_str(document).unwrap();
        let limits = |max_encoded_bytes| PersistedDocumentLimits {
            max_encoded_bytes,
            validation: DocumentValidationLimits {
                max_container_entries: 4,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            kind: PersistedDocumentKind::Coverage,
        };

        write_manifest_document(root.path(), ordinary);
        CoverageManifest::load_with_limits(&path, limits(ordinary.len())).unwrap();

        write_manifest_document(root.path(), document);
        let loaded = CoverageManifest::load_with_limits(&path, limits(document.len()));

        assert_eq!(expected.version, 1);
        let error = loaded.expect_err("a tagged key wrapper must consume an entry");
        assert!(error.to_string().contains("failed to parse YAML fixture"));
    }

    #[test]
    fn coverage_public_loader_rejects_duplicate_decoded_tagged_mapping_keys() {
        let root = fixture_root();
        for document in [
            "!schema version: 1\nversion: 1\nscope: []\nfeatures: []\n",
            "!first version: 1\n!second version: 1\nscope: []\nfeatures: []\n",
        ] {
            write_manifest_document(root.path(), document);
            assert!(matches!(
                CoverageManifest::load(&root.path().join("coverage.yaml")),
                Err(FixtureError::YamlFixture { .. })
            ));
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the test deliberately asserts every committed fixture field"
    )]
    #[expect(
        clippy::cognitive_complexity,
        reason = "the test deliberately asserts the complete committed coverage graph"
    )]
    #[test]
    fn checked_in_inference_scenarios_match_the_recorded_coverage_manifest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../integration/fixtures/inference");

        let manifest = CoverageManifest::load(&root.join("coverage.yaml")).unwrap();
        let scenarios = discover_scenarios(&root).unwrap();
        let report = check_coverage(&root).unwrap();

        assert_eq!(
            manifest.scope,
            vec![
                "messages_to_chat_completions",
                "messages_native_passthrough",
                "responses_agentic_loop",
                "responses_native_passthrough",
                "responses_to_chat_completions",
            ]
        );
        assert_eq!(
            manifest
                .features
                .iter()
                .map(|feature| feature.scopes.iter().map(String::as_str).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![
                vec!["messages_to_chat_completions"],
                vec!["messages_to_chat_completions"],
                vec!["messages_to_chat_completions"],
                vec!["messages_native_passthrough"],
                vec!["messages_native_passthrough"],
                vec!["messages_native_passthrough"],
                vec!["responses_native_passthrough"],
                vec!["responses_native_passthrough"],
                vec!["responses_native_passthrough"],
                vec!["responses_to_chat_completions"],
                vec!["responses_to_chat_completions"],
                vec!["responses_agentic_loop"],
                vec!["responses_to_chat_completions"],
            ]
        );
        assert_eq!(
            manifest
                .features
                .iter()
                .map(|feature| feature.status.clone())
                .collect::<Vec<_>>(),
            vec![
                CoverageStatus::LiveCovered,
                CoverageStatus::LiveCovered,
                CoverageStatus::SyntheticOnly,
                CoverageStatus::LiveCovered,
                CoverageStatus::LiveCovered,
                CoverageStatus::LiveCovered,
                CoverageStatus::LiveCovered,
                CoverageStatus::LiveCovered,
                CoverageStatus::LiveCovered,
                CoverageStatus::SyntheticOnly,
                CoverageStatus::SyntheticOnly,
                CoverageStatus::SyntheticOnly,
                CoverageStatus::SyntheticOnly,
            ]
        );
        assert_eq!(report.features_total, 13);
        assert_eq!(report.scenarios_total, 11);
        assert_eq!(report.recordings_total, 16);
        assert_eq!(
            scenarios.keys().collect::<Vec<_>>(),
            vec![
                "messages/basic-nonstream",
                "messages/basic-stream",
                "messages/native-basic-nonstream",
                "messages/native-basic-stream",
                "messages/native-tool-use",
                "messages/upstream-error",
                "responses/agentic-parallel-tool-calls",
                "responses/chat-basic-nonstream",
                "responses/native-basic-nonstream",
                "responses/native-basic-stream",
                "responses/native-tool-call",
            ]
        );
        assert_eq!(manifest.features.len(), 13);
        assert_eq!(manifest.version, 1);
        assert_eq!(
            manifest
                .features
                .iter()
                .map(|feature| (&feature.id, &feature.scenarios))
                .collect::<Vec<_>>(),
            vec![
                (
                    &"messages.request.minimal".to_owned(),
                    &vec![
                        "messages/basic-nonstream".to_owned(),
                        "messages/basic-stream".to_owned()
                    ]
                ),
                (
                    &"messages.response.text".to_owned(),
                    &vec![
                        "messages/basic-nonstream".to_owned(),
                        "messages/basic-stream".to_owned()
                    ]
                ),
                (
                    &"messages.error.upstream".to_owned(),
                    &vec!["messages/upstream-error".to_owned()]
                ),
                (
                    &"messages.native.request".to_owned(),
                    &vec![
                        "messages/native-basic-nonstream".to_owned(),
                        "messages/native-basic-stream".to_owned(),
                        "messages/native-tool-use".to_owned(),
                    ]
                ),
                (
                    &"messages.native.response.text".to_owned(),
                    &vec![
                        "messages/native-basic-nonstream".to_owned(),
                        "messages/native-basic-stream".to_owned(),
                    ]
                ),
                (
                    &"messages.native.tool_use".to_owned(),
                    &vec!["messages/native-tool-use".to_owned()]
                ),
                (
                    &"responses.native.request".to_owned(),
                    &vec![
                        "responses/native-basic-nonstream".to_owned(),
                        "responses/native-basic-stream".to_owned(),
                        "responses/native-tool-call".to_owned(),
                    ]
                ),
                (
                    &"responses.native.response.text".to_owned(),
                    &vec![
                        "responses/native-basic-nonstream".to_owned(),
                        "responses/native-basic-stream".to_owned(),
                    ]
                ),
                (
                    &"responses.native.tool_call".to_owned(),
                    &vec!["responses/native-tool-call".to_owned()]
                ),
                (
                    &"responses.chat.request".to_owned(),
                    &vec!["responses/chat-basic-nonstream".to_owned()]
                ),
                (
                    &"responses.chat.response.text".to_owned(),
                    &vec!["responses/chat-basic-nonstream".to_owned()]
                ),
                (
                    &"responses.agentic.parallel_tool_calls".to_owned(),
                    &vec!["responses/agentic-parallel-tool-calls".to_owned()]
                ),
                (
                    &"responses.chat.continuation".to_owned(),
                    &vec!["responses/chat-basic-nonstream".to_owned()]
                ),
            ]
        );
        assert_eq!(
            manifest.features[0]
                .providers
                .iter()
                .map(|(provider, coverage)| (provider.as_str(), coverage.status.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("openai", CoverageStatus::Covered),
                ("vllm", CoverageStatus::LiveCovered),
            ]
        );
        assert_eq!(
            manifest.features[1]
                .providers
                .iter()
                .map(|(provider, coverage)| (provider.as_str(), coverage.status.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("openai", CoverageStatus::Covered),
                ("vllm", CoverageStatus::LiveCovered),
            ]
        );
        assert_eq!(
            manifest.features[2]
                .providers
                .iter()
                .map(|(provider, coverage)| (provider.as_str(), coverage.status.clone()))
                .collect::<Vec<_>>(),
            vec![("synthetic", CoverageStatus::SyntheticOnly)]
        );
        for feature in &manifest.features[3..6] {
            assert_eq!(
                feature
                    .providers
                    .iter()
                    .map(|(provider, coverage)| (provider.as_str(), coverage.status.clone()))
                    .collect::<Vec<_>>(),
                vec![("anthropic", CoverageStatus::LiveCovered)]
            );
        }
        for feature in &manifest.features[6..9] {
            assert_eq!(
                feature
                    .providers
                    .iter()
                    .map(|(provider, coverage)| (provider.as_str(), coverage.status.clone()))
                    .collect::<Vec<_>>(),
                vec![
                    ("openai", CoverageStatus::LiveCovered),
                    ("vllm", CoverageStatus::LiveCovered),
                ]
            );
        }
        for feature in &manifest.features[9..] {
            assert_eq!(
                feature
                    .providers
                    .iter()
                    .map(|(provider, coverage)| (provider.as_str(), coverage.status.clone()))
                    .collect::<Vec<_>>(),
                vec![("synthetic", CoverageStatus::SyntheticOnly)]
            );
        }
        assert!(manifest.features.iter().all(|feature| {
            feature.reason.is_none() && feature.providers.values().all(|coverage| coverage.reason.is_none())
        }));

        let nonstream = InferenceScenario::load(&root.join("scenarios/messages/basic-nonstream.yaml")).unwrap();
        assert_scenario(
            &nonstream,
            "messages/basic-nonstream",
            "Minimal Anthropic Messages request translated to Chat Completions.",
            &["messages.request.minimal", "messages.response.text"],
            "What is 2+2? Reply with just the number.",
            false,
            BodyKind::Json,
            200,
            &[],
        );
        let stream = InferenceScenario::load(&root.join("scenarios/messages/basic-stream.yaml")).unwrap();
        assert_scenario(
            &stream,
            "messages/basic-stream",
            "Minimal streaming Anthropic Messages request translated to Chat Completions.",
            &["messages.request.minimal", "messages.response.text"],
            "Say hello in one sentence.",
            true,
            BodyKind::Sse,
            200,
            &[
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        );
        let error = InferenceScenario::load(&root.join("scenarios/messages/upstream-error.yaml")).unwrap();
        assert_scenario(
            &error,
            "messages/upstream-error",
            "Upstream rate-limit error translated for an Anthropic Messages client.",
            &["messages.error.upstream"],
            "What is 2+2? Reply with just the number.",
            false,
            BodyKind::Json,
            429,
            &[],
        );

        let native_nonstream =
            InferenceScenario::load(&root.join("scenarios/messages/native-basic-nonstream.yaml")).unwrap();
        let native_stream = InferenceScenario::load(&root.join("scenarios/messages/native-basic-stream.yaml")).unwrap();
        let native_tool = InferenceScenario::load(&root.join("scenarios/messages/native-tool-use.yaml")).unwrap();
        for scenario in [&native_nonstream, &native_stream, &native_tool] {
            assert_eq!(scenario.example_config, "anthropic/messages-protocol.yaml");
            assert_eq!(scenario.upstream_authority, "127.0.0.1:3001");
            assert_eq!(scenario.turns[0].request.path, "/v1/messages");
            assert_eq!(scenario.turns[0].expect.upstream_path, "/v1/messages");
        }
        assert_eq!(native_nonstream.turns[0].expect.client_body_kind, BodyKind::Json);
        assert_eq!(native_tool.turns[0].expect.client_body_kind, BodyKind::Json);
        assert_eq!(native_stream.turns[0].expect.client_body_kind, BodyKind::Sse);
        assert_eq!(native_stream.turns[0].expect.client_sse_interleaved_events, ["ping"]);
        assert_eq!(native_stream.turns[0].expect.upstream_sse_interleaved_events, ["ping"]);

        let responses_nonstream =
            InferenceScenario::load(&root.join("scenarios/responses/native-basic-nonstream.yaml")).unwrap();
        let responses_stream =
            InferenceScenario::load(&root.join("scenarios/responses/native-basic-stream.yaml")).unwrap();
        let responses_tool = InferenceScenario::load(&root.join("scenarios/responses/native-tool-call.yaml")).unwrap();
        for scenario in [&responses_nonstream, &responses_stream, &responses_tool] {
            assert_eq!(scenario.version, 1);
            assert_eq!(scenario.protocol, InferenceProtocol::OpenaiResponses);
            assert_eq!(scenario.example_config, "openai/responses/responses-proxy.yaml");
            assert_eq!(scenario.upstream_authority, "127.0.0.1:3001");
            assert_eq!(scenario.turns.len(), 1);
            let turn = &scenario.turns[0];
            assert_eq!(turn.name, "initial");
            assert_eq!(turn.request.method, "POST");
            assert_eq!(turn.request.path, "/v1/responses");
            assert_eq!(
                turn.request.headers.get("content-type"),
                Some(&vec!["application/json".to_owned()])
            );
            assert_eq!(turn.expect.client_status, 200);
            assert_eq!(turn.expect.upstream_path, "/v1/responses");
            assert_eq!(turn.expect.upstream_body_kind, BodyKind::Json);
            let RecordedBody::Json { value } = &turn.request.body else {
                panic!("native Responses request body must be JSON");
            };
            assert_eq!(value["model"], "${MODEL}");
            assert_eq!(value["store"], false);
        }
        assert_eq!(responses_nonstream.id, "responses/native-basic-nonstream");
        assert_eq!(
            responses_nonstream.description,
            "Minimal non-streaming OpenAI Responses request passed through to a native backend."
        );
        assert_eq!(
            responses_nonstream.features,
            ["responses.native.request", "responses.native.response.text"]
        );
        let RecordedBody::Json { value } = &responses_nonstream.turns[0].request.body else {
            panic!("native non-streaming Responses request body must be JSON");
        };
        assert_eq!(value["input"], "Reply with exactly `4` and nothing else.");
        assert_eq!(value["max_output_tokens"], 16);
        assert_eq!(value["stream"], false);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(5));
        assert_eq!(responses_nonstream.turns[0].expect.client_body_kind, BodyKind::Json);
        assert!(responses_nonstream.turns[0].expect.client_sse_events.is_empty());
        assert!(responses_nonstream.turns[0].expect.upstream_sse_events.is_empty());

        assert_eq!(responses_stream.id, "responses/native-basic-stream");
        assert_eq!(
            responses_stream.description,
            "Minimal streaming OpenAI Responses request passed through to a native backend."
        );
        assert_eq!(
            responses_stream.features,
            ["responses.native.request", "responses.native.response.text"]
        );
        let RecordedBody::Json { value } = &responses_stream.turns[0].request.body else {
            panic!("native streaming Responses request body must be JSON");
        };
        assert_eq!(value["input"], "Reply with exactly `4` and nothing else.");
        assert_eq!(value["max_output_tokens"], 16);
        assert_eq!(value["stream"], true);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(5));
        assert_eq!(responses_stream.turns[0].expect.client_body_kind, BodyKind::Sse);
        assert_eq!(
            responses_stream.turns[0].expect.client_sse_events,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(
            responses_stream.turns[0].expect.client_sse_repeatable_events,
            ["response.output_text.delta"]
        );
        assert_eq!(
            responses_stream.turns[0].expect.upstream_sse_events,
            responses_stream.turns[0].expect.client_sse_events
        );
        assert_eq!(
            responses_stream.turns[0].expect.upstream_sse_repeatable_events,
            ["response.output_text.delta"]
        );
        assert!(
            responses_stream.turns[0]
                .expect
                .client_sse_interleaved_events
                .is_empty()
        );
        assert!(
            responses_stream.turns[0]
                .expect
                .upstream_sse_interleaved_events
                .is_empty()
        );

        assert_eq!(responses_tool.id, "responses/native-tool-call");
        assert_eq!(
            responses_tool.description,
            "Forced OpenAI Responses function call passed through to a native backend."
        );
        assert_eq!(
            responses_tool.features,
            ["responses.native.request", "responses.native.tool_call"]
        );
        let RecordedBody::Json { value } = &responses_tool.turns[0].request.body else {
            panic!("native tool-call Responses request body must be JSON");
        };
        assert_eq!(
            value["input"],
            "What is the weather in Boston? Use the get_weather function."
        );
        assert_eq!(value["max_output_tokens"], 64);
        assert_eq!(value["stream"], false);
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["name"], "get_weather");
        assert_eq!(value["tool_choice"], json!({"type": "function", "name": "get_weather"}));
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(7));
        assert_eq!(responses_tool.turns[0].expect.client_body_kind, BodyKind::Json);
        assert!(responses_tool.turns[0].expect.client_sse_events.is_empty());
        assert!(responses_tool.turns[0].expect.upstream_sse_events.is_empty());

        let responses_chat =
            InferenceScenario::load(&root.join("scenarios/responses/chat-basic-nonstream.yaml")).unwrap();
        assert_eq!(responses_chat.version, 1);
        assert_eq!(responses_chat.id, "responses/chat-basic-nonstream");
        assert_eq!(
            responses_chat.description,
            "Finite OpenAI Responses request translated to Chat Completions."
        );
        assert_eq!(responses_chat.protocol, InferenceProtocol::OpenaiResponses);
        assert_eq!(
            responses_chat.example_config,
            "openai/responses/responses-to-chat-completions.yaml"
        );
        assert_eq!(responses_chat.upstream_authority, "127.0.0.1:3001");
        assert_eq!(
            responses_chat.features,
            [
                "responses.chat.request",
                "responses.chat.response.text",
                "responses.chat.continuation",
            ]
        );
        assert_eq!(responses_chat.turns.len(), 2);
        let turn = &responses_chat.turns[0];
        assert_eq!(turn.name, "initial");
        assert_eq!(turn.request.method, "POST");
        assert_eq!(turn.request.path, "/v1/responses");
        assert_eq!(turn.expect.client_status, 200);
        assert_eq!(turn.expect.client_body_kind, BodyKind::Json);
        assert_eq!(turn.expect.upstream_path, "/v1/chat/completions");
        assert_eq!(turn.expect.upstream_body_kind, BodyKind::Json);
        let RecordedBody::Json { value } = &turn.request.body else {
            panic!("translated Responses request body must be JSON");
        };
        assert_eq!(value["model"], "${MODEL}");
        assert_eq!(value["input"], "What is 2+2? Reply with just the number.");
        assert_eq!(value["store"], true);
        assert_eq!(value["stream"], false);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(4));
        assert!(
            turn.expect.client_sse_events.is_empty(),
            "non-streaming initial turn must have no client SSE events"
        );
        assert!(
            turn.expect.upstream_sse_events.is_empty(),
            "non-streaming initial turn must have no upstream SSE events"
        );

        let continuation = &responses_chat.turns[1];
        assert_eq!(continuation.name, "continuation");
        assert_eq!(continuation.request.path, "/v1/responses");
        assert_eq!(continuation.expect.client_status, 200);
        assert_eq!(continuation.expect.client_body_kind, BodyKind::Json);
        assert_eq!(continuation.expect.upstream_path, "/v1/chat/completions");
        assert_eq!(continuation.expect.upstream_body_kind, BodyKind::Json);
        let RecordedBody::Json { value } = &continuation.request.body else {
            panic!("continuation Responses request body must be JSON");
        };
        assert_eq!(value["model"], "${MODEL}");
        assert_eq!(
            value["input"],
            "What was the previous answer? Reply with just the number."
        );
        assert_eq!(value["previous_response_id"], "${PREVIOUS_RESPONSE_ID}");
        assert_eq!(value["store"], false);
        assert_eq!(value["stream"], false);
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(5));
        assert!(
            continuation.expect.client_sse_events.is_empty(),
            "non-streaming continuation turn must have no client SSE events"
        );
        assert!(
            continuation.expect.upstream_sse_events.is_empty(),
            "non-streaming continuation turn must have no upstream SSE events"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_files_and_directories() {
        use std::os::unix::fs::symlink;

        let root = fixture_root();
        fs::write(root.path().join("outside.yaml"), "version: 1\nid: case/one\n").unwrap();
        symlink(
            root.path().join("outside.yaml"),
            root.path().join("scenarios/link.yaml"),
        )
        .unwrap();
        assert!(discover_scenarios(root.path()).is_err());

        let root = fixture_root();
        fs::create_dir_all(root.path().join("outside")).unwrap();
        symlink(root.path().join("outside"), root.path().join("recordings/link")).unwrap();
        assert!(discover_recordings(root.path()).is_err());
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the helper mirrors the independent scenario assertion contract"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "the helper verifies every externally consumed scenario field"
    )]
    #[expect(
        clippy::cognitive_complexity,
        reason = "the explicit optional-stream assertion keeps fixture checks legible"
    )]
    fn assert_scenario(
        scenario: &InferenceScenario,
        id: &str,
        description: &str,
        features: &[&str],
        prompt: &str,
        stream: bool,
        body_kind: BodyKind,
        status: u16,
        events: &[&str],
    ) {
        assert_eq!(scenario.version, 1);
        assert_eq!(scenario.id, id);
        assert_eq!(scenario.description, description);
        assert_eq!(scenario.protocol, InferenceProtocol::AnthropicMessages);
        assert_eq!(scenario.example_config, "anthropic/messages-to-openai.yaml");
        assert_eq!(scenario.upstream_authority, "127.0.0.1:8000");
        assert_eq!(scenario.features, features);
        assert_eq!(scenario.turns.len(), 1);
        let turn = &scenario.turns[0];
        assert_eq!(turn.name, "initial");
        assert_eq!(turn.request.method, "POST");
        assert_eq!(turn.request.path, "/v1/messages");
        assert_eq!(
            turn.request.headers.get("content-type"),
            Some(&vec!["application/json".to_owned()])
        );
        let RecordedBody::Json { value } = &turn.request.body else {
            panic!("request body must be JSON");
        };
        assert_eq!(value["model"], "${MODEL}");
        assert_eq!(value["max_tokens"], 64);
        if stream {
            assert_eq!(value["stream"], true);
        } else {
            assert_eq!(value["stream"], false);
        }
        assert_eq!(value.as_object().map(serde_json::Map::len), Some(4));
        assert_eq!(value["messages"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"], prompt);
        assert_eq!(turn.expect.client_status, status);
        assert_eq!(turn.expect.client_body_kind, body_kind);
        assert_eq!(turn.expect.upstream_path, "/v1/chat/completions");
        assert_eq!(turn.expect.upstream_body_kind, BodyKind::Json);
        assert_eq!(turn.expect.client_sse_events, events);
        assert_eq!(
            turn.expect.client_sse_repeatable_events,
            if stream {
                vec!["content_block_delta".to_owned()]
            } else {
                Vec::new()
            }
        );
        assert!(turn.expect.upstream_sse_events.is_empty());
        assert!(turn.expect.upstream_sse_repeatable_events.is_empty());
    }

    #[test]
    fn status_matrix_requires_evidence_for_every_linked_scenario() {
        for (status, recordings) in [
            ("covered", vec![]),
            ("live_covered", vec![("source_a", "imported")]),
            ("synthetic_only", vec![("source_a", "live")]),
            ("unsupported", vec![("source_a", "imported")]),
        ] {
            let root = fixture_root();
            write_manifest_document(
                root.path(),
                &format!(
                    "version: 1\nscope: [messages_to_chat_completions]\nfeatures:\n  - id: feature.one\n    scopes: [messages_to_chat_completions]\n    status: {status}\n    reason: reviewed\n    scenarios: [case/one, case/two]\n    providers:\n      source_a: {{status: covered}}\n"
                ),
            );
            write_scenario_at(root.path(), "one.yaml", "case/one", &["feature.one"]);
            write_scenario_at(root.path(), "two.yaml", "case/two", &["feature.one"]);
            for (provider, kind) in recordings {
                write_recording(root.path(), "one.json", "case/one", provider, kind);
            }

            assert!(
                check_coverage(root.path()).is_err(),
                "{status} accepted insufficient evidence"
            );
        }
    }

    #[test]
    fn provider_matrix_rejects_wrong_provenance_and_unsupported_live_evidence() {
        for (provider_status, kind) in [
            ("covered", "synthetic"),
            ("live_covered", "imported"),
            ("synthetic_only", "live"),
            ("provider_unsupported", "live"),
        ] {
            let root = fixture_root();
            write_manifest_document(
                root.path(),
                &format!(
                    "version: 1\nscope: [messages_to_chat_completions]\nfeatures:\n  - id: feature.one\n    scopes: [messages_to_chat_completions]\n    status: covered\n    scenarios: [case/one]\n    providers:\n      source_a:\n        status: {provider_status}\n        reason: reviewed\n"
                ),
            );
            write_scenario(root.path(), "case/one", &["feature.one"]);
            write_recording(root.path(), "one.json", "case/one", "source_a", kind);

            assert!(
                check_coverage(root.path()).is_err(),
                "{provider_status}/{kind} was accepted"
            );
        }
    }

    #[test]
    fn rejects_undeclared_live_evidence_when_provider_policy_is_declared() {
        let root = fixture_root();
        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one]\n    providers:\n      unavailable: {status: provider_unsupported, reason: absent}\n",
        );
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "undeclared.json", "case/one", "other", "live");

        assert!(check_coverage(root.path()).is_err());
    }

    #[test]
    fn unsupported_provider_entries_allow_zero_evidence_but_reject_contradictions() {
        for status in ["unsupported", "provider_unsupported"] {
            let root = fixture_root();
            write_manifest_document(
                root.path(),
                &format!(
                    "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: synthetic_only\n    scenarios: [case/one]\n    providers:\n      unavailable: {{status: {status}, reason: absent}}\n"
                ),
            );
            write_scenario(root.path(), "case/one", &["feature.one"]);
            write_recording(root.path(), "synthetic.json", "case/one", "fixture", "synthetic");
            assert!(check_coverage(root.path()).is_ok());

            write_recording(root.path(), "contradiction.json", "case/one", "unavailable", "live");
            assert!(check_coverage(root.path()).is_err());
        }
    }

    #[test]
    fn report_counts_multiple_statuses_without_duplicate_inflation() {
        let root = fixture_root();
        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: covered.feature\n    scopes: [messages]\n    status: covered\n    scenarios: [case/covered]\n  - id: live.feature\n    scopes: [messages]\n    status: live_covered\n    scenarios: [case/live]\n    providers:\n      source_a: {status: covered}\n  - id: synthetic.feature\n    scopes: [messages]\n    status: synthetic_only\n    scenarios: [case/synthetic]\n",
        );
        write_scenario_at(root.path(), "covered.yaml", "case/covered", &["covered.feature"]);
        write_scenario_at(root.path(), "live.yaml", "case/live", &["live.feature"]);
        write_scenario_at(root.path(), "synthetic.yaml", "case/synthetic", &["synthetic.feature"]);
        write_recording(root.path(), "covered.json", "case/covered", "source_b", "imported");
        write_recording(root.path(), "live.json", "case/live", "source_a", "live");
        write_recording(root.path(), "synthetic.json", "case/synthetic", "fixture", "synthetic");

        let report = check_coverage(root.path()).unwrap();
        assert_eq!(
            (report.features_total, report.scenarios_total, report.recordings_total),
            (3, 3, 3)
        );
        assert_eq!(
            report.counts_by_status,
            std::collections::BTreeMap::from([
                (CoverageStatus::Covered, 1),
                (CoverageStatus::LiveCovered, 1),
                (CoverageStatus::SyntheticOnly, 1),
            ])
        );
    }

    #[test]
    fn report_does_not_inflate_shared_scenario_or_recording_totals() {
        let root = fixture_root();
        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/shared]\n  - id: feature.two\n    scopes: [messages]\n    status: covered\n    scenarios: [case/shared]\n",
        );
        write_scenario(root.path(), "case/shared", &["feature.one", "feature.two"]);
        write_recording(root.path(), "shared.json", "case/shared", "source", "imported");

        let report = check_coverage(root.path()).unwrap();
        assert_eq!(
            (report.features_total, report.scenarios_total, report.recordings_total),
            (2, 1, 1)
        );
        assert_eq!(
            report.counts_by_status,
            std::collections::BTreeMap::from([(CoverageStatus::Covered, 2)])
        );
    }

    #[test]
    fn provider_unsupported_accepts_synthetic_or_supported_alternate_provider_evidence() {
        let root = fixture_root();
        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages_to_chat_completions]\nfeatures:\n  - id: feature.one\n    scopes: [messages_to_chat_completions]\n    status: provider_unsupported\n    reason: source_a lacks this capability\n    scenarios: [case/one]\n    providers:\n      source_a: {status: provider_unsupported, reason: capability absent}\n      source_b: {status: covered}\n",
        );
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "alternate.json", "case/one", "source_b", "imported");

        assert!(check_coverage(root.path()).is_ok());
    }

    #[test]
    fn provider_unsupported_requires_declarations_and_rejects_undeclared_alternates() {
        let root = fixture_root();
        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: provider_unsupported\n    reason: reviewed\n    scenarios: [case/one]\n",
        );
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "undeclared.json", "case/one", "other", "live");
        assert!(check_coverage(root.path()).is_err());

        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: provider_unsupported\n    reason: reviewed\n    scenarios: [case/one]\n    providers:\n      unavailable: {status: provider_unsupported, reason: absent}\n",
        );
        fs::remove_file(root.path().join("recordings/undeclared.json")).unwrap();
        write_recording(root.path(), "synthetic.json", "case/one", "fixture", "synthetic");
        assert!(check_coverage(root.path()).is_ok());

        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: provider_unsupported\n    reason: reviewed\n    scenarios: [case/one]\n    providers:\n      unavailable: {status: provider_unsupported, reason: absent}\n      alternate: {status: covered}\n",
        );
        fs::remove_file(root.path().join("recordings/synthetic.json")).unwrap();
        write_recording(root.path(), "alternate.json", "case/one", "alternate", "imported");
        assert!(check_coverage(root.path()).is_ok());
        fs::remove_file(root.path().join("recordings/alternate.json")).unwrap();
        write_recording(root.path(), "undeclared.json", "case/one", "other", "live");
        assert!(check_coverage(root.path()).is_err());
    }

    #[test]
    fn rejects_invalid_scope_feature_links() {
        // Catches one-way, absent, and duplicate scope membership in the feature denominator.
        for document in [
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: []\n    status: covered\n    scenarios: [case/one]\n",
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [undeclared]\n    status: covered\n    scenarios: [case/one]\n",
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages, messages]\n    status: covered\n    scenarios: [case/one]\n",
            "version: 1\nscope: [messages, unused]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one]\n",
        ] {
            let root = fixture_root();
            write_manifest_document(root.path(), document);
            write_scenario(root.path(), "case/one", &["feature.one"]);
            write_recording(root.path(), "one.json", "case/one", "source", "imported");

            assert!(
                check_coverage(root.path()).is_err(),
                "accepted invalid manifest:\n{document}"
            );
        }

        let root = fixture_root();
        write_manifest_document(
            root.path(),
            "version: 1\nscope: [messages, shared]\nfeatures:\n  - id: feature.one\n    scopes: [messages, shared]\n    status: covered\n    scenarios: [case/one]\n",
        );
        write_scenario(root.path(), "case/one", &["feature.one"]);
        write_recording(root.path(), "one.json", "case/one", "source", "imported");

        assert!(check_coverage(root.path()).is_ok());
    }

    #[test]
    fn rejects_empty_and_duplicate_coverage_graph_identifiers() {
        for document in [
            "version: 1\nscope: []\nfeatures: []\n",
            "version: 1\nscope: [messages, messages]\nfeatures: []\n",
            "version: 1\nscope: [messages]\nfeatures:\n  - id: ' '\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one]\n",
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: []\n",
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one, case/one]\n",
            "version: 1\nscope: [messages]\nfeatures:\n  - id: feature.one\n    scopes: [messages]\n    status: covered\n    scenarios: [case/one]\n    providers:\n      ' ': {status: covered}\n",
        ] {
            let root = fixture_root();
            write_manifest_document(root.path(), document);
            write_scenario(root.path(), "case/one", &["feature.one"]);
            assert!(check_coverage(root.path()).is_err());
        }

        let root = fixture_root();
        write_manifest(root.path(), "covered", "feature.one", "case/one", "");
        write_scenario(root.path(), "case/one", &["feature.one", "feature.one"]);
        assert!(check_coverage(root.path()).is_err());
    }

    #[test]
    fn rejects_duplicate_recording_content_identity_but_allows_distinct_providers() {
        let root = fixture_root();
        write_recording(root.path(), "a.json", "case/one", "source_a", "live");
        write_recording(root.path(), "b.json", "case/one", "source_a", "synthetic");
        let error = discover_recordings(root.path()).unwrap_err();
        assert!(error.to_string().contains("duplicate recording identity"));
        assert!(error.to_string().contains("a.json"));
        assert!(error.to_string().contains("b.json"));

        let root = fixture_root();
        write_recording(root.path(), "a.json", "case/one", "source_a", "live");
        write_recording(root.path(), "b.json", "case/one", "source_b", "live");
        assert_eq!(discover_recordings(root.path()).unwrap().len(), 2);
    }

    fn write_manifest_document(root: &Path, document: &str) {
        fs::write(root.join("coverage.yaml"), document).unwrap();
    }

    fn fixture_root() -> TempDir {
        let root = TempDir::new().unwrap();
        fs::create_dir_all(root.path().join("scenarios")).unwrap();
        fs::create_dir_all(root.path().join("recordings")).unwrap();
        root
    }

    fn write_manifest(root: &Path, status: &str, feature: &str, scenario: &str, extra: &str) {
        let extra = if extra.is_empty() {
            String::new()
        } else {
            format!(
                "\n{}",
                extra
                    .lines()
                    .map(|line| format!("    {line}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        fs::write(
            root.join("coverage.yaml"),
            format!(
                "version: 1\nscope:\n  - messages_to_chat_completions\nfeatures:\n  - id: {feature}\n    scopes:\n      - messages_to_chat_completions\n    status: {status}\n    scenarios:\n      - {scenario}{extra}\n"
            ),
        )
        .unwrap();
    }

    fn write_scenario(root: &Path, id: &str, features: &[&str]) {
        write_scenario_at(root, "scenario.yaml", id, features);
    }

    fn write_scenario_at(root: &Path, relative_path: &str, id: &str, features: &[&str]) {
        let path = root.join("scenarios").join(relative_path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let features = features
            .iter()
            .map(|feature| format!("  - {feature}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(
            path,
            format!(
                "version: 1\nid: {id}\ndescription: test scenario\nprotocol: anthropic_messages\nexample_config: example.yaml\nupstream_authority: localhost:8000\nfeatures:\n{features}\nturns:\n  - name: initial\n    request:\n      method: POST\n      path: /v1/messages\n      body:\n        kind: json\n        value: {{}}\n    expect:\n      client_status: 200\n      client_body_kind: json\n      upstream_path: /v1/chat/completions\n      upstream_body_kind: json\n"
            ),
        )
        .unwrap();
    }

    fn write_recording(root: &Path, relative_path: &str, scenario_id: &str, provider: &str, kind: &str) {
        let path = root.join("recordings").join(relative_path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            path,
            serde_json::to_vec_pretty(&json!({
                "version": 1,
                "scenario_id": scenario_id,
                "protocol": "anthropic_messages",
                "provenance": {"kind": kind, "provider": provider, "model": "commit-safe-model", "source_id": null},
                "normalization": {"version": 1, "linked_ids": {}},
                "turns": [{
                    "name": "initial",
                    "client": {
                        "request": {"method": "POST", "path": "/", "headers": {}, "body": {"kind": "empty"}},
                        "response": {"status": 204, "headers": {}, "body": {"kind": "empty"}}
                    },
                    "upstream": {
                        "request": {"method": "POST", "path": "/", "headers": {}, "body": {"kind": "empty"}},
                        "response": {"status": 204, "headers": {}, "body": {"kind": "empty"}}
                    }
                }]
            }))
            .unwrap(),
        )
        .unwrap();
    }
}
