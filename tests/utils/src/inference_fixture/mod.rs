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

// -----------------------------------------------------------------------------
// HTTP Client Construction
// -----------------------------------------------------------------------------

/// A reqwest client on the installed crypto provider.
///
/// reqwest is built with `rustls-no-provider`, so constructing a client
/// before the process-wide provider is installed panics inside rustls;
/// every construction in this module goes through here (or
/// [`http_client_builder`]), which installs the OpenSSL-backed provider
/// first, as the binary does at startup. Only this module's own tests build
/// bare clients; library code always configures a builder.
#[cfg(test)]
pub(crate) fn http_client() -> reqwest::Client {
    crate::net::tls::ensure_crypto_provider();
    reqwest::Client::new()
}

/// A reqwest client builder on the installed crypto provider; see
/// `http_client` above.
pub(crate) fn http_client_builder() -> reqwest::ClientBuilder {
    crate::net::tls::ensure_crypto_provider();
    reqwest::Client::builder()
}

/// Default maximum request body bytes an ad-hoc HTTP recorder should retain.
///
/// Mirrors the replay scenario request ceiling so a wire tap and the fixture
/// recorder agree on one bound.
pub const MAX_RECORDED_REQUEST_BODY_BYTES: usize = bounds::MAX_SCENARIO_REQUEST_BODY_BYTES;

/// Default maximum response body bytes an ad-hoc HTTP recorder should retain.
///
/// Mirrors the scripted response ceiling used by the fixture recorder.
pub const MAX_RECORDED_RESPONSE_BODY_BYTES: usize = bounds::MAX_SCRIPTED_RESPONSE_BODY_BYTES;

/// Parses one captured HTTP body into the shared [`RecordedBody`] representation.
///
/// Reuses the inference recorder's content-type dispatch, JSON structural
/// preflight, and SSE frame-count and frame-size bounds so ad-hoc recorders
/// (such as a wire tap) never introduce a second body representation.
///
/// # Errors
///
/// Returns a [`FixtureError`] when the body is not valid for its content type
/// or exceeds the shared structural or SSE bounds.
pub fn record_http_body(content_type: Option<&str>, bytes: &[u8]) -> Result<RecordedBody, FixtureError> {
    bounds::parse_response_body(content_type, bytes)
}

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
