// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Versioned wire fixtures and inference scenarios for integration tests.

mod bounds;
mod coverage;
mod external;
mod header_policy;
mod http_server;
mod record;
mod replay;
mod sanitize;
mod schema;

pub use coverage::{
    CoverageFeature, CoverageManifest, CoverageReport, CoverageStatus, ProviderCoverage, RecordingRef,
    ScenarioSnapshot, check_coverage, discover_recordings, discover_scenario_snapshots, discover_scenarios,
};
pub use external::{ImportedUpstream, import_external_recording};
pub use record::{ProviderTarget, RecordingProxy, RecordingProxyGuard};
pub use replay::{ReplayReport, ScenarioRunner};
pub use sanitize::{RedactionRules, sanitize_fixture, validate_commit_safe, validate_commit_safe_with_rules};
pub use schema::{
    BodyKind, FixtureError, FixtureProvenance, INFERENCE_SCENARIO_VERSION, InferenceProtocol, InferenceScenario,
    MAX_INFERENCE_TURNS, NORMALIZATION_VERSION, NormalizationMetadata, ProvenanceKind, RecordedBody, RecordedExchange,
    RecordedRequest, RecordedResponse, ScenarioExpectation, ScenarioTurn, SseFrame, WIRE_FIXTURE_VERSION, WireFixture,
    WireTurn,
};
