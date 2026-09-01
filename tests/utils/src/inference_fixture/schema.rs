// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Versioned fixture and scenario schema implementation.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File},
    io::{Read, Write as _},
    path::{Path, PathBuf},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeOwned, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, VariantAccess as _, Visitor},
};
use serde_json::Value;
use thiserror::Error;

/// The supported version of the recorded wire-fixture format.
pub const WIRE_FIXTURE_VERSION: u32 = 1;

/// The supported version of fixture normalization metadata.
pub const NORMALIZATION_VERSION: u32 = 1;

/// The supported version of the inference-scenario format.
pub const INFERENCE_SCENARIO_VERSION: u32 = 1;

/// Maximum ordered turns accepted in one scenario or wire fixture.
pub const MAX_INFERENCE_TURNS: usize = 64;

/// Maximum canonical bytes in one recorded request body.
const MAX_REQUEST_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Maximum canonical bytes in one recorded response body.
const MAX_RESPONSE_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Sum of the two 8 MiB requests and two 16 MiB responses in one wire turn.
#[cfg(test)]
const MAX_ONE_TURN_WIRE_BODY_BYTES: usize = 48 * 1024 * 1024;

/// Worst-case standard-Base64 text for all four bounded bodies in one turn.
const MAX_BASE64_ONE_TURN_BODY_BYTES: usize =
    base64_encoded_len(MAX_REQUEST_BODY_BYTES) * 2 + base64_encoded_len(MAX_RESPONSE_BODY_BYTES) * 2;

/// Maximum encoded size of one loaded two-sided wire fixture.
///
/// This leaves another full one-turn Base64 budget for JSON/schema overhead and
/// ordinary multi-turn fixtures. Larger committed documents fail explicitly.
pub(super) const MAX_FIXTURE_DOCUMENT_BYTES: usize = MAX_BASE64_ONE_TURN_BODY_BYTES * 2;

/// Maximum encoded size of one provider-neutral scenario document.
///
/// This permits two independently Base64-encoded maximum 8 MiB requests plus
/// schema overhead while remaining far below the two-sided fixture ceiling.
const MAX_SCENARIO_DOCUMENT_BYTES: usize = 32 * 1024 * 1024;

/// Maximum encoded size of the body-free coverage manifest.
pub(super) const MAX_COVERAGE_DOCUMENT_BYTES: usize = 1024 * 1024;

/// Maximum values observed before typed wire-fixture deserialization.
pub(super) const MAX_WIRE_FIXTURE_NODES: usize = 1_000_000;

/// Maximum sequence elements plus mapping entries in one wire fixture.
pub(super) const MAX_WIRE_FIXTURE_CONTAINER_ENTRIES: usize = 1_000_000;

/// Maximum decoded bytes across wire-fixture keys and string values.
pub(super) const MAX_WIRE_FIXTURE_DECODED_STRING_BYTES: usize = MAX_FIXTURE_DOCUMENT_BYTES;

/// Maximum nested sequence/mapping depth in one wire fixture.
pub(super) const MAX_WIRE_FIXTURE_CONTAINER_DEPTH: usize = 128;

/// Maximum values in one scenario document before typed materialization.
const MAX_SCENARIO_NODES: usize = 250_000;

/// Maximum sequence elements plus mapping entries in one scenario document.
const MAX_SCENARIO_CONTAINER_ENTRIES: usize = 250_000;

/// Maximum values in one coverage manifest before typed materialization.
const MAX_COVERAGE_NODES: usize = 20_000;

/// Maximum sequence elements plus mapping entries in one coverage manifest.
const MAX_COVERAGE_CONTAINER_ENTRIES: usize = 20_000;

/// Maximum decoded key and string bytes in one coverage manifest.
const MAX_COVERAGE_DECODED_STRING_BYTES: usize = 512 * 1024;

/// Maximum mapping/sequence nesting in one coverage manifest.
const MAX_COVERAGE_CONTAINER_DEPTH: usize = 32;

/// Returns the padded standard-Base64 length for one independently encoded body.
const fn base64_encoded_len(raw_len: usize) -> usize {
    raw_len.div_ceil(3) * 4
}

/// A versioned, normalized recording of client and upstream exchanges.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WireFixture {
    /// The wire-fixture schema version.
    pub version: u32,
    /// The identifier of the scenario that uses this fixture.
    pub scenario_id: String,
    /// The inference protocol represented by the fixture.
    pub protocol: InferenceProtocol,
    /// Metadata describing where the recording came from.
    pub provenance: FixtureProvenance,
    /// Metadata describing normalizations applied to the recording.
    pub normalization: NormalizationMetadata,
    /// Ordered request and response exchanges in the recording.
    pub turns: Vec<WireTurn>,
}

/// An ordered client and upstream exchange in a wire fixture.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WireTurn {
    /// A stable, human-readable name for this exchange.
    pub name: String,
    /// The request and response observed by the fixture client.
    pub client: RecordedExchange,
    /// The request and response observed by the fixture upstream.
    pub upstream: RecordedExchange,
}

/// A recorded request and response pair.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedExchange {
    /// The recorded request.
    pub request: RecordedRequest,
    /// The recorded response.
    pub response: RecordedResponse,
}

/// A recorded HTTP request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedRequest {
    /// The HTTP method.
    pub method: String,
    /// The request path, including any query string.
    pub path: String,
    /// Header values, grouped by header name.
    #[serde(default)]
    pub headers: BTreeMap<String, Vec<String>>,
    /// The request body in a portable fixture representation.
    pub body: RecordedBody,
}

/// A recorded HTTP response.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedResponse {
    /// The HTTP response status code.
    pub status: u16,
    /// Header values, grouped by header name.
    #[serde(default)]
    pub headers: BTreeMap<String, Vec<String>>,
    /// The response body in a portable fixture representation.
    pub body: RecordedBody,
}

/// A portable representation of an HTTP body.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecordedBody {
    /// An empty body.
    Empty,
    /// A body parsed as JSON.
    Json {
        /// The parsed JSON value.
        value: Value,
    },
    /// A body parsed as server-sent event frames.
    Sse {
        /// The parsed event frames, excluding the terminal `[DONE]` marker.
        frames: Vec<SseFrame>,
        /// Whether the stream included a terminal `[DONE]` marker.
        done: bool,
    },
    /// An arbitrary body encoded using standard Base64.
    Base64 {
        /// The standard-Base64 encoded body bytes.
        data: String,
    },
}

/// Strict deserialization representation for every recorded-body variant.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StrictRecordedBody {
    /// An empty body with no additional fields.
    Empty {},
    /// A body parsed as JSON.
    Json {
        /// The parsed JSON value.
        value: Value,
    },
    /// A body parsed as server-sent event frames.
    Sse {
        /// The parsed event frames.
        frames: Vec<SseFrame>,
        /// Whether the terminal marker was present.
        done: bool,
    },
    /// An arbitrary body encoded using standard Base64.
    Base64 {
        /// The standard-Base64 encoded body bytes.
        data: String,
    },
}

impl<'de> Deserialize<'de> for RecordedBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match StrictRecordedBody::deserialize(deserializer)? {
            StrictRecordedBody::Empty {} => Self::Empty,
            StrictRecordedBody::Json { value } => Self::Json { value },
            StrictRecordedBody::Sse { frames, done } => Self::Sse { frames, done },
            StrictRecordedBody::Base64 { data } => Self::Base64 { data },
        })
    }
}

/// One server-sent event frame.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SseFrame {
    /// The optional SSE event name.
    pub event: Option<String>,
    /// The frame's data lines joined with newline characters.
    pub data: String,
    /// The optional SSE event identifier.
    pub id: Option<String>,
    /// The optional SSE reconnection delay in milliseconds.
    pub retry: Option<u64>,
}

/// The inference API protocol represented by a fixture or scenario.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceProtocol {
    /// The OpenAI Responses API.
    OpenaiResponses,
    /// The Anthropic Messages API.
    AnthropicMessages,
    /// The OpenAI Chat Completions API.
    OpenaiChatCompletions,
}

/// The source category for a fixture recording.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    /// A recording captured from a live provider interaction.
    Live,
    /// A recording imported from another source.
    Imported,
    /// A fixture written without a provider recording.
    Synthetic,
}

/// Metadata describing the source of a fixture recording.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureProvenance {
    /// The source category.
    pub kind: ProvenanceKind,
    /// The provider that supplied or inspired the recording.
    pub provider: String,
    /// The provider model used for the recording.
    pub model: String,
    /// An optional provider or import source identifier.
    pub source_id: Option<String>,
}

/// Metadata about normalizations applied to a fixture recording.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizationMetadata {
    /// The normalization schema version.
    pub version: u32,
    /// Mappings from original identifiers to normalized identifiers.
    pub linked_ids: BTreeMap<String, String>,
}

/// A versioned scenario that drives an inference fixture.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceScenario {
    /// The inference-scenario schema version.
    pub version: u32,
    /// A stable scenario identifier.
    pub id: String,
    /// A human-readable explanation of the scenario.
    pub description: String,
    /// The inference protocol exercised by the scenario.
    pub protocol: InferenceProtocol,
    /// The example configuration used to run the scenario.
    pub example_config: String,
    /// The authority expected for upstream requests.
    pub upstream_authority: String,
    /// Features exercised by this scenario.
    pub features: Vec<String>,
    /// Ordered requests and expectations in the scenario.
    pub turns: Vec<ScenarioTurn>,
}

/// A request and its expected outcomes in an inference scenario.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioTurn {
    /// A stable, human-readable name for this turn.
    pub name: String,
    /// The request sent to the scenario client.
    pub request: RecordedRequest,
    /// The expected client and upstream outcomes.
    pub expect: ScenarioExpectation,
}

impl ScenarioTurn {
    /// Binds exact `${PREVIOUS_RESPONSE_ID}` JSON values to the immediately
    /// preceding client response identifier.
    pub(super) fn bind_previous_response_id(&mut self, previous_response_id: Option<&str>) -> Result<(), FixtureError> {
        let RecordedBody::Json { value } = &mut self.request.body else {
            return Ok(());
        };
        let has_placeholder = contains_exact_string(value, "${PREVIOUS_RESPONSE_ID}");
        if !has_placeholder {
            return Ok(());
        }
        let Some(previous_response_id) = previous_response_id else {
            return Err(FixtureError::ReplayRuntime {
                message: "scenario previous response ID placeholder has no preceding response",
            });
        };
        bind_exact_string(value, "${PREVIOUS_RESPONSE_ID}", previous_response_id);
        Ok(())
    }
}

impl RecordedResponse {
    /// Returns the top-level Responses resource identifier, when present.
    pub(super) fn response_id(&self) -> Option<&str> {
        let RecordedBody::Json { value } = &self.body else {
            return None;
        };
        value
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| id.starts_with("resp_"))
    }
}

/// The expected observable outcomes for a scenario turn.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioExpectation {
    /// The expected client response status code.
    pub client_status: u16,
    /// The expected client response body representation.
    pub client_body_kind: BodyKind,
    /// The expected upstream request path.
    pub upstream_path: String,
    /// The expected upstream request body representation.
    pub upstream_body_kind: BodyKind,
    /// The expected client SSE event names.
    #[serde(default)]
    pub client_sse_events: Vec<String>,
    /// Client SSE event names that may repeat contiguously one or more times.
    #[serde(default)]
    pub client_sse_repeatable_events: Vec<String>,
    /// Client SSE event names that may appear anywhere zero or more times.
    #[serde(default)]
    pub client_sse_interleaved_events: Vec<String>,
    /// The expected upstream SSE event names.
    #[serde(default)]
    pub upstream_sse_events: Vec<String>,
    /// Upstream SSE event names that may repeat contiguously one or more times.
    #[serde(default)]
    pub upstream_sse_repeatable_events: Vec<String>,
    /// Upstream SSE event names that may appear anywhere zero or more times.
    #[serde(default)]
    pub upstream_sse_interleaved_events: Vec<String>,
}

/// The representation used for a recorded body.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyKind {
    /// An empty body.
    Empty,
    /// A parsed JSON body.
    Json,
    /// A parsed server-sent event body.
    Sse,
    /// A Base64-encoded binary body.
    Base64,
}

/// Errors produced while loading, writing, or rendering inference fixtures.
#[derive(Debug, Error)]
pub enum FixtureError {
    /// The fixture file could not be read.
    #[error("failed to read fixture file `{path}`")]
    Read {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A persisted document exceeded its dedicated bounded loader ceiling.
    #[error("persisted {kind} document exceeds its byte limit")]
    PersistedDocumentTooLarge {
        /// Static document category; never derived from fixture content.
        kind: &'static str,
    },
    /// The exact persisted-document buffer could not be allocated.
    #[error("persisted fixture document allocation failed")]
    PersistedDocumentAllocation,
    /// A fixture document changed length while its open handle was being read.
    #[error("fixture document changed while it was being read")]
    FixtureDocumentChanged,
    /// The fixture file could not be written.
    #[error("failed to write fixture file `{path}`")]
    Write {
        /// The file that could not be written.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A JSON fixture document could not be parsed.
    #[error("failed to parse JSON fixture file `{path}`")]
    JsonFixture {
        /// The JSON fixture file that could not be parsed.
        path: PathBuf,
        /// The underlying JSON parsing error.
        #[source]
        source: serde_json::Error,
    },
    /// A YAML fixture document could not be parsed.
    #[error("failed to parse YAML fixture file `{path}`")]
    YamlFixture {
        /// The YAML fixture file that could not be parsed.
        path: PathBuf,
        /// The underlying YAML parsing error.
        #[source]
        source: serde_yaml::Error,
    },
    /// A JSON body could not be parsed.
    #[error("invalid JSON body")]
    JsonBodyParse(#[source] serde_json::Error),
    /// A JSON body could not be rendered.
    #[error("failed to render JSON body")]
    JsonBodyRender(#[source] serde_json::Error),
    /// An external request-hashed recording did not match the supported envelope.
    #[error("invalid external recording: {message}")]
    ExternalRecording {
        /// A non-secret explanation of the invalid envelope shape.
        message: &'static str,
    },
    /// An external request-hashed recording could not be parsed as JSON.
    #[error("invalid external recording JSON")]
    ExternalRecordingJson(#[source] serde_json::Error),
    /// A legacy recording omitted both supported response representations.
    #[error("legacy recording is missing a response body")]
    LegacyRecordingMissingResponse,
    /// A JSON fixture document could not be serialized.
    #[error("failed to serialize JSON fixture file `{path}`")]
    JsonFixtureSerialization {
        /// The JSON fixture file that could not be serialized.
        path: PathBuf,
        /// The underlying JSON serialization error.
        #[source]
        source: serde_json::Error,
    },
    /// A YAML fixture document could not be serialized.
    #[error("failed to serialize YAML fixture file `{path}`")]
    YamlFixtureSerialization {
        /// The YAML fixture file that could not be serialized.
        path: PathBuf,
        /// The underlying YAML serialization error.
        #[source]
        source: serde_yaml::Error,
    },
    /// An SSE body was not valid UTF-8.
    #[error("invalid UTF-8 SSE body")]
    SseUtf8(#[source] std::str::Utf8Error),
    /// A Base64 body could not be decoded.
    #[error("invalid Base64 body")]
    Base64(#[source] base64::DecodeError),
    /// The fixture contains data that is unsafe to commit.
    #[error("fixture commit safety violation: {rule} at {path}")]
    CommitSafety {
        /// The safety rule that was violated.
        rule: &'static str,
        /// The JSON path where the rule was detected.
        path: String,
    },
    /// The deterministic identifier sequence was exhausted.
    #[error("fixture identifier normalization sequence exhausted")]
    NormalizationIdOverflow,
    /// A literal redaction rule did not name any source text.
    #[error("redaction rule contains an empty literal source")]
    EmptyRedactionLiteral,
    /// A wire fixture has an unsupported version.
    #[error("unsupported wire fixture version {version}; expected {WIRE_FIXTURE_VERSION}")]
    UnsupportedWireFixtureVersion {
        /// The unsupported fixture version.
        version: u32,
    },
    /// A wire fixture has an unsupported normalization metadata version.
    #[error("unsupported fixture normalization version")]
    UnsupportedNormalizationVersion {
        /// The unsupported normalization version, retained for typed handling.
        version: u32,
    },
    /// An inference scenario has an unsupported version.
    #[error("unsupported inference scenario version {version}; expected {INFERENCE_SCENARIO_VERSION}")]
    UnsupportedInferenceScenarioVersion {
        /// The unsupported scenario version.
        version: u32,
    },
    /// A scenario or wire fixture declared a vacuous or excessive turn count.
    #[error("{document} turn count {count} is outside 1..={MAX_INFERENCE_TURNS}")]
    InvalidInferenceTurnCount {
        /// Stable schema document category.
        document: &'static str,
        /// Rejected number of ordered turns.
        count: usize,
    },
    /// An inference scenario declared an invalid SSE expectation pattern.
    #[error("invalid scenario expectation at {path}: {rule}")]
    InvalidScenarioExpectation {
        /// Deterministic path to the invalid expectation field.
        path: String,
        /// Opaque rule that the expectation violated.
        rule: &'static str,
    },
    /// A coverage manifest has an unsupported version.
    #[error("unsupported coverage manifest version {version}; expected 1")]
    UnsupportedCoverageManifestVersion {
        /// The unsupported manifest version.
        version: u32,
    },
    /// A directory required by fixture discovery could not be read.
    #[error("failed to read fixture directory `{path}`")]
    ReadDirectory {
        /// The directory that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Discovery encountered a symlink, which it deliberately never follows.
    #[error("fixture discovery does not allow symlink `{path}`")]
    FixtureSymlink {
        /// The symlink path.
        path: PathBuf,
    },
    /// Multiple scenario files declared the same stable identifier.
    #[error("duplicate scenario ID `{id}`")]
    DuplicateScenarioId {
        /// The duplicate identifier.
        id: String,
    },
    /// Multiple manifest features declared the same stable identifier.
    #[error("duplicate coverage feature ID `{id}`")]
    DuplicateCoverageFeatureId {
        /// The duplicate identifier.
        id: String,
    },
    /// A scenario named a feature that the manifest does not declare.
    #[error("scenario `{scenario_id}` references unknown feature `{feature_id}`")]
    CoverageUnknownFeature {
        /// The scenario that named the feature.
        scenario_id: String,
        /// The unknown feature identifier.
        feature_id: String,
    },
    /// A feature named a scenario for which no scenario file exists.
    #[error("feature `{feature_id}` references missing scenario `{scenario_id}`")]
    CoverageMissingScenario {
        /// The feature that named the scenario.
        feature_id: String,
        /// The missing scenario identifier.
        scenario_id: String,
    },
    /// A feature/scenario link was declared on only one side of the graph.
    #[error("feature `{feature_id}` and scenario `{scenario_id}` are not linked bidirectionally")]
    CoverageOneSidedLink {
        /// The feature identifier.
        feature_id: String,
        /// The scenario identifier.
        scenario_id: String,
    },
    /// A scenario does not declare any coverage features.
    #[error("scenario `{scenario_id}` must reference at least one feature")]
    CoverageScenarioWithoutFeatures {
        /// The scenario identifier.
        scenario_id: String,
    },
    /// An unsupported coverage entry omitted its required explanation.
    #[error("feature `{feature_id}` requires a nonempty reason for status {status}")]
    CoverageReasonRequired {
        /// The feature identifier.
        feature_id: String,
        /// The unsupported status.
        status: String,
    },
    /// An unsupported provider coverage entry omitted its required explanation.
    #[error("feature `{feature_id}` provider `{provider}` requires a nonempty reason for status {status}")]
    CoverageProviderReasonRequired {
        /// The feature identifier.
        feature_id: String,
        /// The provider identifier.
        provider: String,
        /// The unsupported status.
        status: String,
    },
    /// A provider-unsupported feature did not identify any providers.
    #[error("feature `{feature_id}` with provider_unsupported status must declare a provider")]
    CoverageProviderRequired {
        /// The feature identifier.
        feature_id: String,
    },
    /// A live-covered feature had no live recording from a declared provider.
    #[error("feature `{feature_id}` requires a live recording from a declared provider")]
    CoverageLiveRecordingRequired {
        /// The feature identifier.
        feature_id: String,
    },
    /// A synthetic-only feature had no synthetic recording for one of its scenarios.
    #[error("feature `{feature_id}` requires a synthetic recording")]
    CoverageSyntheticRecordingRequired {
        /// The feature identifier.
        feature_id: String,
    },
    /// A recording declared a scenario that does not exist in the fixture tree.
    #[error("recording `{path}` declares unknown scenario `{scenario_id}`")]
    CoverageUnknownRecordingScenario {
        /// The recording path.
        path: PathBuf,
        /// The unknown scenario identifier.
        scenario_id: String,
    },
    /// A recording and its matched scenario declared different inference protocols.
    #[error("recording `{path}` protocol does not match scenario `{scenario_id}`")]
    CoverageRecordingProtocolMismatch {
        /// The recording path.
        path: PathBuf,
        /// The matched scenario identifier.
        scenario_id: String,
    },
    /// A coverage-manifest invariant was violated.
    #[error("invalid coverage manifest: {message}")]
    CoverageInvariant {
        /// A non-secret explanation of the invalid invariant.
        message: String,
    },
    /// Two recording files declared the same scenario and provider identity.
    #[error(
        "duplicate recording identity for scenario `{scenario_id}` and provider `{provider}`: `{first_path}` and `{second_path}`"
    )]
    DuplicateRecordingIdentity {
        /// The duplicated scenario identifier.
        scenario_id: String,
        /// The duplicated provider identifier.
        provider: String,
        /// First deterministic recording path.
        first_path: PathBuf,
        /// Second deterministic recording path.
        second_path: PathBuf,
    },
    /// Replay infrastructure could not complete an opaque runtime operation.
    #[error("{message}")]
    ReplayRuntime {
        /// A non-secret description of the failed replay operation.
        message: &'static str,
    },
    /// Replay client I/O failed without retaining transport details.
    #[error("scenario HTTP operation failed")]
    ReplayHttp,
    /// A normalized materialization or replay comparison differed.
    #[error("replay mismatch at {path}: {rule}")]
    ReplayMismatch {
        /// Deterministic fixture or scenario path that differed.
        path: String,
        /// Non-secret rule describing the kind of mismatch.
        rule: &'static str,
    },
}

/// One strict-loaded fixture plus raw-document safety state for the checker.
pub(super) struct LoadedWireFixture {
    /// Typed fixture consumed by replay and coverage validation.
    pub(super) fixture: WireFixture,
    /// First default-policy violation found while streaming decoded JSON text.
    pub(super) raw_commit_safety_error: Option<FixtureError>,
}

impl WireFixture {
    /// Loads and validates a wire fixture from a JSON or YAML file.
    ///
    /// Files whose extension is `.json` are parsed as JSON; all other files
    /// are parsed as YAML.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounded file buffer cannot be read or allocated,
    /// strict structural validation or typed parsing fails, the fixture or
    /// normalization version is unsupported, or its turn count is outside the
    /// shared bound.
    pub fn load(path: &Path) -> Result<Self, FixtureError> {
        Ok(Self::load_for_commit_check(path)?.fixture)
    }

    /// Strict-loads a fixture while retaining raw decoded-text safety state.
    pub(super) fn load_for_commit_check(path: &Path) -> Result<LoadedWireFixture, FixtureError> {
        let parsed: ParsedDocument<Self> = load_persisted_document(path, PersistedDocumentLimits::WIRE_FIXTURE)?;
        let fixture = parsed.value;
        let raw_commit_safety_error = parsed
            .validation
            .commit_safety_rule
            .map(|rule| FixtureError::CommitSafety {
                rule,
                path: "$/<raw>".to_owned(),
            });
        fixture.validate_version()?;
        Ok(LoadedWireFixture {
            fixture,
            raw_commit_safety_error,
        })
    }

    /// Writes this wire fixture as JSON or YAML based on the file extension.
    ///
    /// Files whose extension is `.json` are written as JSON; all other files
    /// are written as YAML.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::UnsupportedWireFixtureVersion`] if the fixture
    /// schema version is unsupported,
    /// [`FixtureError::UnsupportedNormalizationVersion`] if the normalization
    /// metadata version is unsupported, [`FixtureError::InvalidInferenceTurnCount`]
    /// if its turn count is outside the shared bound,
    /// [`FixtureError::PersistedDocumentTooLarge`] if serialization exceeds the
    /// loader ceiling, or an error if serialization or writing fails.
    pub fn write(&self, path: &Path) -> Result<(), FixtureError> {
        let document = match fixture_format(path) {
            FixtureFormat::Json => self.to_pretty_json_document(path)?,
            FixtureFormat::Yaml => serialize_yaml_with_limit(self, path, MAX_FIXTURE_DOCUMENT_BYTES)?,
        };
        fs::write(path, document).map_err(|source| FixtureError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Serializes one loader-compatible pretty-JSON document with a final newline.
    ///
    /// `diagnostic_path` is used only in an opaque serialization error and is
    /// never read or written by this method.
    ///
    /// # Errors
    ///
    /// Returns a schema-validation error, [`FixtureError::PersistedDocumentTooLarge`]
    /// at the loader byte ceiling, [`FixtureError::PersistedDocumentAllocation`]
    /// if the bounded buffer cannot grow, or an opaque JSON serialization error.
    pub fn to_pretty_json_document(&self, diagnostic_path: &Path) -> Result<Vec<u8>, FixtureError> {
        serialize_pretty_json_with_limit(self, diagnostic_path, MAX_FIXTURE_DOCUMENT_BYTES)
    }

    /// Verifies the wire-fixture and normalization versions and ordered turn-count bound.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::UnsupportedWireFixtureVersion`] when `version`
    /// is not [`WIRE_FIXTURE_VERSION`], or
    /// [`FixtureError::UnsupportedNormalizationVersion`] when
    /// `normalization.version` is not [`NORMALIZATION_VERSION`], or
    /// [`FixtureError::InvalidInferenceTurnCount`] when `turns` is outside
    /// `1..=`[`MAX_INFERENCE_TURNS`].
    pub fn validate_version(&self) -> Result<(), FixtureError> {
        if self.version != WIRE_FIXTURE_VERSION {
            return Err(FixtureError::UnsupportedWireFixtureVersion { version: self.version });
        }
        if self.normalization.version != NORMALIZATION_VERSION {
            return Err(FixtureError::UnsupportedNormalizationVersion {
                version: self.normalization.version,
            });
        }
        validate_turn_count("wire fixture", self.turns.len())
    }
}

/// Why one bounded serialization buffer refused another write.
#[derive(Clone, Copy)]
enum DocumentWriteFailure {
    /// The next write would exceed the encoded document ceiling.
    TooLarge,
    /// The fallible buffer reservation failed.
    Allocation,
}

/// Fallibly accumulates at most one persisted document ceiling.
struct BoundedDocumentWriter {
    /// Serialized bytes accepted so far.
    document: Vec<u8>,
    /// Maximum complete encoded document bytes.
    limit: usize,
    /// First bounded write failure, if any.
    failure: Option<DocumentWriteFailure>,
}

impl BoundedDocumentWriter {
    /// Creates an empty bounded serialization buffer.
    const fn new(limit: usize) -> Self {
        Self {
            document: Vec::new(),
            limit,
            failure: None,
        }
    }

    /// Converts a bounded write failure to the shared loader vocabulary.
    fn fixture_error(&self) -> Option<FixtureError> {
        self.failure.map(|failure| match failure {
            DocumentWriteFailure::TooLarge => FixtureError::PersistedDocumentTooLarge { kind: "wire fixture" },
            DocumentWriteFailure::Allocation => FixtureError::PersistedDocumentAllocation,
        })
    }

    /// Returns the serialized document after successful completion.
    fn into_inner(self) -> Vec<u8> {
        self.document
    }

    /// Grows geometrically without asking `Vec` to reserve beyond the limit.
    fn reserve_for(&mut self, next_len: usize) -> Result<(), DocumentWriteFailure> {
        if next_len <= self.document.capacity() {
            return Ok(());
        }
        let target_capacity = self
            .document
            .capacity()
            .max(8 * 1024)
            .saturating_mul(2)
            .max(next_len)
            .min(self.limit);
        self.document
            .try_reserve_exact(target_capacity.saturating_sub(self.document.len()))
            .map_err(|_source| DocumentWriteFailure::Allocation)
    }
}

impl std::io::Write for BoundedDocumentWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let Some(next_len) = self.document.len().checked_add(buf.len()) else {
            self.failure = Some(DocumentWriteFailure::TooLarge);
            return Err(std::io::Error::other("fixture document size overflowed"));
        };
        if next_len > self.limit {
            self.failure = Some(DocumentWriteFailure::TooLarge);
            return Err(std::io::Error::other("fixture document exceeded limit"));
        }
        if let Err(failure) = self.reserve_for(next_len) {
            self.failure = Some(failure);
            return Err(std::io::Error::other("fixture document allocation failed"));
        }
        self.document.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Serializes bounded pretty JSON and accounts for its canonical final newline.
fn serialize_pretty_json_with_limit(
    fixture: &WireFixture,
    diagnostic_path: &Path,
    limit: usize,
) -> Result<Vec<u8>, FixtureError> {
    serialize_pretty_json_with_limits(fixture, diagnostic_path, limit, DocumentValidationLimits::WIRE_FIXTURE)
}

/// Serializes bounded pretty JSON under explicit structural test/production limits.
fn serialize_pretty_json_with_limits(
    fixture: &WireFixture,
    diagnostic_path: &Path,
    limit: usize,
    structural_limits: DocumentValidationLimits,
) -> Result<Vec<u8>, FixtureError> {
    fixture.validate_version()?;
    let mut writer = BoundedDocumentWriter::new(limit);
    if let Err(source) = serde_json::to_writer_pretty(&mut writer, fixture) {
        return Err(writer
            .fixture_error()
            .unwrap_or_else(|| FixtureError::JsonFixtureSerialization {
                path: diagnostic_path.to_path_buf(),
                source,
            }));
    }
    if writer.write_all(b"\n").is_err() {
        return Err(writer
            .fixture_error()
            .unwrap_or(FixtureError::PersistedDocumentAllocation));
    }
    validate_json_document_with_limits(&writer.document, structural_limits).map_err(|source| {
        FixtureError::JsonFixtureSerialization {
            path: diagnostic_path.to_path_buf(),
            source,
        }
    })?;
    Ok(writer.into_inner())
}

/// Serializes bounded YAML through the same encoded loader ceiling.
fn serialize_yaml_with_limit(
    fixture: &WireFixture,
    diagnostic_path: &Path,
    limit: usize,
) -> Result<Vec<u8>, FixtureError> {
    serialize_yaml_with_limits(fixture, diagnostic_path, limit, DocumentValidationLimits::WIRE_FIXTURE)
}

/// Serializes bounded YAML under explicit structural test/production limits.
fn serialize_yaml_with_limits(
    fixture: &WireFixture,
    diagnostic_path: &Path,
    limit: usize,
    structural_limits: DocumentValidationLimits,
) -> Result<Vec<u8>, FixtureError> {
    fixture.validate_version()?;
    preflight_wire_fixture_for_yaml(fixture, diagnostic_path, MAX_FIXTURE_DOCUMENT_BYTES, structural_limits)?;
    let mut writer = BoundedDocumentWriter::new(limit);
    if let Err(source) = serde_yaml::to_writer(&mut writer, fixture) {
        return Err(writer
            .fixture_error()
            .unwrap_or_else(|| FixtureError::YamlFixtureSerialization {
                path: diagnostic_path.to_path_buf(),
                source,
            }));
    }
    validate_yaml_document_with_limits(&writer.document, structural_limits).map_err(|source| {
        FixtureError::YamlFixtureSerialization {
            path: diagnostic_path.to_path_buf(),
            source,
        }
    })?;
    Ok(writer.into_inner())
}

/// Bounds typed string and structural resources before libyaml copies scalars.
fn preflight_wire_fixture_for_yaml(
    fixture: &WireFixture,
    diagnostic_path: &Path,
    encoded_limit: usize,
    structural_limits: DocumentValidationLimits,
) -> Result<(), FixtureError> {
    let mut writer = BoundedDocumentWriter::new(encoded_limit);
    if let Err(source) = serde_json::to_writer(&mut writer, fixture) {
        return Err(writer
            .fixture_error()
            .unwrap_or_else(|| FixtureError::JsonFixtureSerialization {
                path: diagnostic_path.to_path_buf(),
                source,
            }));
    }
    validate_json_document_with_limits(&writer.document, structural_limits).map_err(|_source| {
        FixtureError::YamlFixtureSerialization {
            path: diagnostic_path.to_path_buf(),
            source: <serde_yaml::Error as serde::ser::Error>::custom("fixture structural limit exceeded"),
        }
    })?;
    Ok(())
}

impl InferenceScenario {
    /// Loads and validates an inference scenario from a JSON or YAML file.
    ///
    /// Files whose extension is `.json` are parsed as JSON; all other files
    /// are parsed as YAML.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounded file buffer cannot be read or allocated,
    /// strict structural validation or typed parsing fails, the scenario
    /// version is unsupported, its turn count is outside the shared bound, or
    /// a request exceeds its canonical body limit.
    pub fn load(path: &Path) -> Result<Self, FixtureError> {
        load_scenario_with_limits(
            path,
            PersistedDocumentLimits::SCENARIO,
            super::bounds::MAX_SCENARIO_REQUEST_BODY_BYTES,
        )
    }

    /// Verifies the scenario version, turn count, and SSE expectation metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::UnsupportedInferenceScenarioVersion`] when
    /// `version` is not [`INFERENCE_SCENARIO_VERSION`], or
    /// [`FixtureError::InvalidInferenceTurnCount`] when `turns` is outside
    /// `1..=`[`MAX_INFERENCE_TURNS`], or
    /// [`FixtureError::InvalidScenarioExpectation`] when an SSE event name is
    /// empty; repeatable SSE event metadata is duplicate, unknown, or
    /// ambiguous; or an interleaved SSE event declaration is empty, duplicated,
    /// or not disjoint from the ordered and repeatable declarations.
    pub fn validate_version(&self) -> Result<(), FixtureError> {
        if self.version != INFERENCE_SCENARIO_VERSION {
            return Err(FixtureError::UnsupportedInferenceScenarioVersion { version: self.version });
        }
        validate_turn_count("inference scenario", self.turns.len())?;
        for (turn_index, turn) in self.turns.iter().enumerate() {
            validate_sse_expectation_lists(
                &turn.expect.client_sse_events,
                &turn.expect.client_sse_repeatable_events,
                &turn.expect.client_sse_interleaved_events,
                &format!("turns[{turn_index}].expect.client_sse"),
            )?;
            validate_sse_expectation_lists(
                &turn.expect.upstream_sse_events,
                &turn.expect.upstream_sse_repeatable_events,
                &turn.expect.upstream_sse_interleaved_events,
                &format!("turns[{turn_index}].expect.upstream_sse"),
            )?;
        }
        Ok(())
    }

    /// Returns a copy of this scenario with exact `${MODEL}` JSON values bound.
    ///
    /// String values merely containing `${MODEL}` are left unchanged.
    #[must_use]
    pub fn bind_model(&self, model: &str) -> Self {
        let mut bound = self.clone();
        for turn in &mut bound.turns {
            if let RecordedBody::Json { value } = &mut turn.request.body {
                bind_model_value(value, model);
            }
        }
        bound
    }
}

/// Requires one meaningful, operationally bounded ordered turn list.
fn validate_turn_count(document: &'static str, count: usize) -> Result<(), FixtureError> {
    if (1..=MAX_INFERENCE_TURNS).contains(&count) {
        Ok(())
    } else {
        Err(FixtureError::InvalidInferenceTurnCount { document, count })
    }
}

/// Loads one scenario under explicit document and canonical request-body limits.
fn load_scenario_with_limits(
    path: &Path,
    document_limits: PersistedDocumentLimits,
    max_request_body_bytes: usize,
) -> Result<InferenceScenario, FixtureError> {
    let scenario: InferenceScenario = load_persisted_document(path, document_limits)?.value;
    scenario.validate_version()?;
    for turn in &scenario.turns {
        super::bounds::validate_request_body_with_limit(&turn.request.body, max_request_body_bytes)?;
    }
    Ok(scenario)
}

/// Rejects empty entries in one client or upstream SSE event-name list.
fn validate_nonempty_sse_event_names(events: &[String], path: &str) -> Result<(), FixtureError> {
    if events.iter().any(String::is_empty) {
        Err(FixtureError::InvalidScenarioExpectation {
            path: path.to_owned(),
            rule: "SSE event name must not be empty",
        })
    } else {
        Ok(())
    }
}

/// Validates the ordered, repeatable, and interleaved declarations for one SSE leg.
fn validate_sse_expectation_lists(
    expected: &[String],
    repeatable: &[String],
    interleaved: &[String],
    path_prefix: &str,
) -> Result<(), FixtureError> {
    validate_nonempty_sse_event_names(expected, &format!("{path_prefix}_events"))?;
    let index = SseDeclarationIndex::new(expected, repeatable);
    validate_repeatable_sse_events(repeatable, &index, &format!("{path_prefix}_repeatable_events"))?;
    validate_interleaved_sse_events(interleaved, &index, &format!("{path_prefix}_interleaved_events"))
}

/// Borrowed occurrence and membership indexes shared by SSE metadata validation.
struct SseDeclarationIndex<'a> {
    /// Number of ordered declarations for each event name.
    ordered_occurrences: BTreeMap<&'a str, usize>,
    /// Event names declared repeatable.
    repeatable: BTreeSet<&'a str>,
}

impl<'a> SseDeclarationIndex<'a> {
    /// Indexes ordered occurrences and repeatable membership without cloning names.
    fn new(expected: &'a [String], repeatable: &'a [String]) -> Self {
        let mut ordered_occurrences = BTreeMap::new();
        for event in expected {
            *ordered_occurrences.entry(event.as_str()).or_insert(0) += 1;
        }
        Self {
            ordered_occurrences,
            repeatable: repeatable.iter().map(String::as_str).collect(),
        }
    }
}

/// Validates one additive one-or-more contiguous SSE repetition declaration.
fn validate_repeatable_sse_events(
    repeatable: &[String],
    index: &SseDeclarationIndex<'_>,
    path: &str,
) -> Result<(), FixtureError> {
    validate_nonempty_sse_event_names(repeatable, path)?;
    let mut declared = BTreeSet::new();
    for event in repeatable {
        if !declared.insert(event.as_str()) {
            return Err(FixtureError::InvalidScenarioExpectation {
                path: path.to_owned(),
                rule: "duplicate repeatable SSE event declaration",
            });
        }
        match index.ordered_occurrences.get(event.as_str()).copied().unwrap_or(0) {
            0 => {
                return Err(FixtureError::InvalidScenarioExpectation {
                    path: path.to_owned(),
                    rule: "repeatable SSE event is not declared in the named-event pattern",
                });
            },
            1 => {},
            _ => {
                return Err(FixtureError::InvalidScenarioExpectation {
                    path: path.to_owned(),
                    rule: "repeatable SSE event declaration is ambiguous",
                });
            },
        }
    }
    Ok(())
}

/// Validates optional named events that may appear between ordered SSE stages.
fn validate_interleaved_sse_events(
    interleaved: &[String],
    index: &SseDeclarationIndex<'_>,
    path: &str,
) -> Result<(), FixtureError> {
    validate_nonempty_sse_event_names(interleaved, path)?;
    let mut declared = BTreeSet::new();
    for event in interleaved {
        if !declared.insert(event.as_str()) {
            return Err(FixtureError::InvalidScenarioExpectation {
                path: path.to_owned(),
                rule: "duplicate interleaved SSE event declaration",
            });
        }
        if index.ordered_occurrences.contains_key(event.as_str()) || index.repeatable.contains(event.as_str()) {
            return Err(FixtureError::InvalidScenarioExpectation {
                path: path.to_owned(),
                rule: "interleaved SSE event overlaps an ordered or repeatable declaration",
            });
        }
    }
    Ok(())
}

impl RecordedBody {
    /// Parses an HTTP body into the portable representation selected by content type.
    ///
    /// Empty bodies become [`RecordedBody::Empty`], JSON media types become
    /// [`RecordedBody::Json`], `text/event-stream` becomes [`RecordedBody::Sse`],
    /// and all other bodies become [`RecordedBody::Base64`].
    ///
    /// # Errors
    ///
    /// Returns an error if JSON cannot be parsed or the SSE body is not UTF-8.
    pub fn from_http(content_type: Option<&str>, bytes: &[u8]) -> Result<Self, FixtureError> {
        Self::from_http_inner(content_type, bytes, None)
    }

    /// Parses an HTTP body while enforcing SSE allocation limits during parsing.
    pub(super) fn from_http_with_sse_limits(
        content_type: Option<&str>,
        bytes: &[u8],
        limits: SseParseLimits,
    ) -> Result<Self, FixtureError> {
        Self::from_http_inner(content_type, bytes, Some(limits))
    }

    /// Shared content-type parser with optional allocation-time SSE limits.
    fn from_http_inner(
        content_type: Option<&str>,
        bytes: &[u8],
        sse_limits: Option<SseParseLimits>,
    ) -> Result<Self, FixtureError> {
        if bytes.is_empty() {
            return Ok(Self::Empty);
        }
        if content_type.is_some_and(is_json_content_type) {
            return serde_json::from_slice(bytes)
                .map(|value| Self::Json { value })
                .map_err(FixtureError::JsonBodyParse);
        }
        if content_type.is_some_and(is_sse_content_type) {
            return parse_sse_body(bytes, sse_limits);
        }
        Ok(Self::Base64 {
            data: STANDARD.encode(bytes),
        })
    }

    /// Renders this portable body representation as HTTP body bytes.
    ///
    /// SSE frames are emitted with canonical LF line endings, and the terminal
    /// `[DONE]` marker is emitted only when it was recorded.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization or Base64 decoding fails.
    pub fn render(&self) -> Result<Vec<u8>, FixtureError> {
        match self {
            Self::Empty => Ok(Vec::new()),
            Self::Json { value } => serde_json::to_vec(value).map_err(FixtureError::JsonBodyRender),
            Self::Sse { frames, done } => Ok(render_sse_body(frames, *done)),
            Self::Base64 { data } => STANDARD.decode(data).map_err(FixtureError::Base64),
        }
    }

    /// Returns the representation kind used by this body.
    #[must_use]
    pub const fn kind(&self) -> BodyKind {
        match self {
            Self::Empty => BodyKind::Empty,
            Self::Json { .. } => BodyKind::Json,
            Self::Sse { .. } => BodyKind::Sse,
            Self::Base64 { .. } => BodyKind::Base64,
        }
    }
}

/// The serialization format selected from a fixture file name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FixtureFormat {
    /// JSON serialization.
    Json,
    /// YAML serialization.
    Yaml,
}

/// Accumulates fields while parsing one SSE event block.
#[derive(Debug, Default)]
struct SseFrameBuilder<'a> {
    /// The event type supplied by the current block.
    event: Option<&'a str>,
    /// The ordered data lines supplied by the current block.
    data_lines: Vec<&'a str>,
    /// The event identifier supplied by the current block.
    id: Option<&'a str>,
    /// The retry delay supplied by the current block.
    retry: Option<u64>,
    /// Canonical field bytes excluding the final blank line.
    field_bytes: usize,
    /// Current canonical event-field bytes.
    event_bytes: usize,
    /// Current canonical id-field bytes.
    id_bytes: usize,
    /// Current canonical retry-field bytes.
    retry_bytes: usize,
}

/// Allocation limits applied while parsing an SSE response.
#[derive(Clone, Copy)]
pub(super) struct SseParseLimits {
    /// Maximum completed non-terminal event count.
    pub(super) max_frames: usize,
    /// Maximum canonical bytes in one completed event.
    pub(super) max_frame_bytes: usize,
}

/// Static category used to select a persisted document's dedicated ceilings.
#[derive(Clone, Copy)]
pub(super) enum PersistedDocumentKind {
    /// A two-sided wire recording.
    WireFixture,
    /// A provider-neutral replay scenario.
    Scenario,
    /// The body-free coverage manifest.
    Coverage,
}

impl PersistedDocumentKind {
    /// Returns the fixed diagnostic label for this document category.
    const fn name(self) -> &'static str {
        match self {
            Self::WireFixture => "wire fixture",
            Self::Scenario => "scenario",
            Self::Coverage => "coverage manifest",
        }
    }
}

/// Encoded and decoded structural ceilings for one persisted document category.
#[derive(Clone, Copy)]
pub(super) struct PersistedDocumentLimits {
    /// Maximum bytes read from the open file handle.
    pub(super) max_encoded_bytes: usize,
    /// Allocation-light limits applied before typed deserialization.
    pub(super) validation: DocumentValidationLimits,
    /// Static category used only for diagnostics.
    pub(super) kind: PersistedDocumentKind,
}

impl PersistedDocumentLimits {
    /// Production limits for the coverage YAML manifest.
    pub(super) const COVERAGE: Self = Self {
        max_encoded_bytes: MAX_COVERAGE_DOCUMENT_BYTES,
        validation: DocumentValidationLimits::COVERAGE,
        kind: PersistedDocumentKind::Coverage,
    };
    /// Production limits for provider-neutral scenario JSON or YAML.
    const SCENARIO: Self = Self {
        max_encoded_bytes: MAX_SCENARIO_DOCUMENT_BYTES,
        validation: DocumentValidationLimits::SCENARIO,
        kind: PersistedDocumentKind::Scenario,
    };
    /// Production limits for two-sided recording JSON or YAML.
    const WIRE_FIXTURE: Self = Self {
        max_encoded_bytes: MAX_FIXTURE_DOCUMENT_BYTES,
        validation: DocumentValidationLimits::WIRE_FIXTURE,
        kind: PersistedDocumentKind::WireFixture,
    };
}

/// Reads a persisted document through one open handle and one fallible buffer.
fn read_fixture_file(path: &Path, limits: PersistedDocumentLimits) -> Result<Vec<u8>, FixtureError> {
    read_fixture_file_with_limits_and_allocator(path, limits, try_allocate_document)
}

/// Reads a persisted document using an injectable bounded allocator.
fn read_fixture_file_with_limits_and_allocator(
    path: &Path,
    limits: PersistedDocumentLimits,
    allocator: impl FnOnce(usize) -> Result<Vec<u8>, ()>,
) -> Result<Vec<u8>, FixtureError> {
    let mut file = File::open(path).map_err(|source| FixtureError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| FixtureError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let encoded_len = usize::try_from(metadata.len()).map_err(|_error| FixtureError::PersistedDocumentTooLarge {
        kind: limits.kind.name(),
    })?;
    if encoded_len > limits.max_encoded_bytes {
        return Err(FixtureError::PersistedDocumentTooLarge {
            kind: limits.kind.name(),
        });
    }
    let document = read_exact_sized_document_with_allocator(&mut file, encoded_len, allocator)
        .map_err(|error| map_exact_document_read_error(path, error))?;
    let final_len = file
        .metadata()
        .map_err(|source| FixtureError::Read {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if final_len != metadata.len() {
        return Err(FixtureError::FixtureDocumentChanged);
    }
    Ok(document)
}

/// Maps the private exact-reader failures into stable fixture errors.
fn map_exact_document_read_error(path: &Path, error: ExactDocumentReadError) -> FixtureError {
    match error {
        ExactDocumentReadError::Io(source) => FixtureError::Read {
            path: path.to_path_buf(),
            source,
        },
        ExactDocumentReadError::LengthChanged => FixtureError::FixtureDocumentChanged,
        ExactDocumentReadError::Allocation => FixtureError::PersistedDocumentAllocation,
    }
}

/// Internal failures from an exact-size reader.
#[derive(Debug)]
enum ExactDocumentReadError {
    /// Underlying I/O failed for a reason other than an observed length change.
    Io(std::io::Error),
    /// The readable byte count did not equal the open handle's declared size.
    LengthChanged,
    /// The explicitly bounded document buffer could not be reserved.
    Allocation,
}

/// Reads exactly `encoded_len` bytes and rejects both shorter and longer input.
#[cfg(test)]
fn read_exact_sized_document(reader: impl Read, encoded_len: usize) -> Result<Vec<u8>, ExactDocumentReadError> {
    read_exact_sized_document_with_allocator(reader, encoded_len, try_allocate_document)
}

/// Exact reader with a small allocator seam for deterministic failure tests.
fn read_exact_sized_document_with_allocator(
    mut reader: impl Read,
    encoded_len: usize,
    allocator: impl FnOnce(usize) -> Result<Vec<u8>, ()>,
) -> Result<Vec<u8>, ExactDocumentReadError> {
    let mut document = allocator(encoded_len).map_err(|()| ExactDocumentReadError::Allocation)?;
    if document.len() != encoded_len {
        return Err(ExactDocumentReadError::Allocation);
    }
    if let Err(source) = reader.read_exact(&mut document) {
        return if source.kind() == std::io::ErrorKind::UnexpectedEof {
            Err(ExactDocumentReadError::LengthChanged)
        } else {
            Err(ExactDocumentReadError::Io(source))
        };
    }
    let mut extra = [0_u8; 1];
    match reader.read(&mut extra) {
        Ok(0) => Ok(document),
        Ok(_) => Err(ExactDocumentReadError::LengthChanged),
        Err(source) => Err(ExactDocumentReadError::Io(source)),
    }
}

/// Fallibly reserves and initializes the exact read buffer without conversion.
fn try_allocate_document(encoded_len: usize) -> Result<Vec<u8>, ()> {
    let mut document = Vec::new();
    document.try_reserve_exact(encoded_len).map_err(|_error| ())?;
    document.resize(encoded_len, 0);
    Ok(document)
}

/// Returns whether an encoded document exceeds the committed-fixture ceiling.
#[cfg(test)]
fn is_fixture_document_oversized(encoded_len: u64) -> bool {
    encoded_len > u64::try_from(MAX_FIXTURE_DOCUMENT_BYTES).unwrap_or(u64::MAX)
}

/// Decoded resource ceilings checked before typed trees are materialized.
#[derive(Clone, Copy)]
pub(super) struct DocumentValidationLimits {
    /// Maximum scalar plus container values.
    pub(super) max_nodes: usize,
    /// Maximum array elements plus object entries.
    pub(super) max_container_entries: usize,
    /// Maximum decoded bytes across keys and string values.
    pub(super) max_decoded_string_bytes: usize,
    /// Maximum nested array/object depth.
    pub(super) max_depth: usize,
}

impl DocumentValidationLimits {
    /// Production ceilings for the body-free coverage manifest.
    const COVERAGE: Self = Self {
        max_nodes: MAX_COVERAGE_NODES,
        max_container_entries: MAX_COVERAGE_CONTAINER_ENTRIES,
        max_decoded_string_bytes: MAX_COVERAGE_DECODED_STRING_BYTES,
        max_depth: MAX_COVERAGE_CONTAINER_DEPTH,
    };
    /// Production ceilings for provider-neutral scenarios.
    const SCENARIO: Self = Self {
        max_nodes: MAX_SCENARIO_NODES,
        max_container_entries: MAX_SCENARIO_CONTAINER_ENTRIES,
        max_decoded_string_bytes: MAX_SCENARIO_DOCUMENT_BYTES,
        max_depth: MAX_WIRE_FIXTURE_CONTAINER_DEPTH,
    };
    /// Permissive small-document limits used as a test baseline.
    #[cfg(test)]
    pub(super) const TEST_PERMISSIVE: Self = Self {
        max_nodes: 100,
        max_container_entries: 100,
        max_decoded_string_bytes: 1_000,
        max_depth: 100,
    };
    /// Production ceilings for persisted wire fixtures.
    const WIRE_FIXTURE: Self = Self {
        max_nodes: MAX_WIRE_FIXTURE_NODES,
        max_container_entries: MAX_WIRE_FIXTURE_CONTAINER_ENTRIES,
        max_decoded_string_bytes: MAX_WIRE_FIXTURE_DECODED_STRING_BYTES,
        max_depth: MAX_WIRE_FIXTURE_CONTAINER_DEPTH,
    };
}

/// Per-leg live structural budget leaves one third of the fixture ceiling for wrappers.
pub(super) const LIVE_CAPTURE_STRUCTURE_LIMITS: DocumentValidationLimits = DocumentValidationLimits {
    max_nodes: MAX_WIRE_FIXTURE_NODES / 3,
    max_container_entries: MAX_WIRE_FIXTURE_CONTAINER_ENTRIES / 3,
    max_decoded_string_bytes: MAX_WIRE_FIXTURE_DECODED_STRING_BYTES / 3,
    max_depth: MAX_WIRE_FIXTURE_CONTAINER_DEPTH - 8,
};

/// Result of one allocation-light streaming document-validation pass.
#[derive(Debug)]
pub(super) struct DocumentValidation {
    /// First generic commit-safety rule found in a decoded key or string.
    commit_safety_rule: Option<&'static str>,
    /// Structural resources consumed by the decoded token stream.
    resources: DocumentResourceUsage,
}

/// Allocation-relevant structural resources consumed by one JSON/YAML value.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DocumentResourceUsage {
    /// Scalar plus container values.
    pub(super) nodes: usize,
    /// Array elements plus object entries.
    pub(super) container_entries: usize,
    /// Decoded bytes across keys and string values.
    pub(super) decoded_string_bytes: usize,
    /// Deepest nested array/object level.
    pub(super) max_depth: usize,
}

/// Typed document plus the safety state observed before materialization.
pub(super) struct ParsedDocument<T> {
    /// Typed document value.
    pub(super) value: T,
    /// Streaming validation result for the original decoded tokens.
    pub(super) validation: DocumentValidation,
}

/// Strictly validates then materializes one bounded persisted document.
pub(super) fn load_persisted_document<T>(
    path: &Path,
    limits: PersistedDocumentLimits,
) -> Result<ParsedDocument<T>, FixtureError>
where
    T: DeserializeOwned,
{
    let document = read_fixture_file(path, limits)?;
    match fixture_format(path) {
        FixtureFormat::Json => parse_strict_json_document_with_limits(&document, path, limits.validation),
        FixtureFormat::Yaml => parse_strict_yaml_document_with_limits(&document, path, limits.validation),
    }
}

/// Parses one JSON document using explicit pre-materialization limits.
fn parse_strict_json_document_with_limits<T>(
    document: &[u8],
    path: &Path,
    limits: DocumentValidationLimits,
) -> Result<ParsedDocument<T>, FixtureError>
where
    T: DeserializeOwned,
{
    let validation =
        validate_json_document_with_limits(document, limits).map_err(|source| FixtureError::JsonFixture {
            path: path.to_path_buf(),
            source,
        })?;
    let value = serde_json::from_slice(document).map_err(|source| FixtureError::JsonFixture {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(ParsedDocument { value, validation })
}

/// Parses one YAML document after strict streaming structural validation.
fn parse_strict_yaml_document_with_limits<T>(
    document: &[u8],
    path: &Path,
    limits: DocumentValidationLimits,
) -> Result<ParsedDocument<T>, FixtureError>
where
    T: DeserializeOwned,
{
    let validation =
        validate_yaml_document_with_limits(document, limits).map_err(|source| FixtureError::YamlFixture {
            path: path.to_path_buf(),
            source,
        })?;
    let value = serde_yaml::from_slice(document).map_err(|source| FixtureError::YamlFixture {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(ParsedDocument { value, validation })
}

/// Validates JSON keys, strings, structure, and trailing input without a tree.
fn validate_json_document_with_limits(
    document: &[u8],
    limits: DocumentValidationLimits,
) -> Result<DocumentValidation, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(document);
    let mut state = DocumentValidationState::new(limits);
    DocumentValueSeed { state: &mut state }.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(DocumentValidation {
        commit_safety_rule: state.commit_safety_rule,
        resources: state.resources(),
    })
}

/// Measures one serialized JSON value under live-capture byte and structural ceilings.
pub(super) fn measure_json_value_with_limits<T: Serialize>(
    value: &T,
    max_encoded_bytes: usize,
    structural_limits: DocumentValidationLimits,
) -> Result<(usize, DocumentResourceUsage), FixtureError> {
    let mut writer = BoundedDocumentWriter::new(max_encoded_bytes);
    serde_json::to_writer(&mut writer, value).map_err(|_source| FixtureError::ReplayRuntime {
        message: "recording capture exceeded aggregate limit",
    })?;
    let validation = validate_json_document_with_limits(&writer.document, structural_limits).map_err(|_source| {
        FixtureError::ReplayRuntime {
            message: "recording capture exceeded structural limit",
        }
    })?;
    Ok((writer.document.len(), validation.resources))
}

/// Preflights raw JSON before `serde_json::Value` can allocate a dense tree.
pub(super) fn validate_json_bytes_with_limits(
    document: &[u8],
    limits: DocumentValidationLimits,
) -> Result<DocumentResourceUsage, FixtureError> {
    validate_json_document_with_limits(document, limits)
        .map(|validation| validation.resources)
        .map_err(FixtureError::JsonBodyParse)
}

/// Validates one YAML document using the same decoded structural visitor.
fn validate_yaml_document_with_limits(
    document: &[u8],
    limits: DocumentValidationLimits,
) -> Result<DocumentValidation, serde_yaml::Error> {
    let mut documents = serde_yaml::Deserializer::from_slice(document);
    let Some(first) = documents.next() else {
        return Err(de::Error::custom("YAML document is empty"));
    };
    let mut state = DocumentValidationState::new(limits);
    DocumentValueSeed { state: &mut state }.deserialize(first)?;
    if documents.next().is_some() {
        return Err(de::Error::custom("multiple YAML documents are not allowed"));
    }
    Ok(DocumentValidation {
        commit_safety_rule: state.commit_safety_rule,
        resources: state.resources(),
    })
}

/// Mutable counters retained only for one streaming validation pass.
struct DocumentValidationState {
    /// Configured resource ceilings.
    limits: DocumentValidationLimits,
    /// Scalar plus container values observed so far.
    nodes: usize,
    /// Array elements plus object entries observed so far.
    container_entries: usize,
    /// Decoded bytes across keys and string values observed so far.
    decoded_string_bytes: usize,
    /// Current nested array/object depth.
    depth: usize,
    /// Deepest nested array/object depth observed.
    max_depth: usize,
    /// First generic commit-safety rule found in decoded text.
    commit_safety_rule: Option<&'static str>,
}

impl DocumentValidationState {
    /// Creates zeroed counters for one document.
    const fn new(limits: DocumentValidationLimits) -> Self {
        Self {
            limits,
            nodes: 0,
            container_entries: 0,
            decoded_string_bytes: 0,
            depth: 0,
            max_depth: 0,
            commit_safety_rule: None,
        }
    }

    /// Snapshots resource counters after a complete value has been consumed.
    const fn resources(&self) -> DocumentResourceUsage {
        DocumentResourceUsage {
            nodes: self.nodes,
            container_entries: self.container_entries,
            decoded_string_bytes: self.decoded_string_bytes,
            max_depth: self.max_depth,
        }
    }

    /// Accounts for one value before Serde dispatches its representation.
    fn consume_node<E: de::Error>(&mut self) -> Result<(), E> {
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| E::custom("document node limit exceeded"))?;
        if self.nodes > self.limits.max_nodes {
            return Err(E::custom("document node limit exceeded"));
        }
        Ok(())
    }

    /// Accounts for one array element or object entry before its value is visited.
    fn consume_container_entry<E: de::Error>(&mut self) -> Result<(), E> {
        self.container_entries = self
            .container_entries
            .checked_add(1)
            .ok_or_else(|| E::custom("document container-entry limit exceeded"))?;
        if self.container_entries > self.limits.max_container_entries {
            return Err(E::custom("document container-entry limit exceeded"));
        }
        Ok(())
    }

    /// Accounts for decoded text without changing the retained safety result.
    fn account_text<E: de::Error>(&mut self, text: &str) -> Result<(), E> {
        self.decoded_string_bytes = self
            .decoded_string_bytes
            .checked_add(text.len())
            .ok_or_else(|| E::custom("document decoded-string limit exceeded"))?;
        if self.decoded_string_bytes > self.limits.max_decoded_string_bytes {
            return Err(E::custom("document decoded-string limit exceeded"));
        }
        Ok(())
    }

    /// Accounts for decoded text and records the first default safety violation.
    fn consume_text<E: de::Error>(&mut self, text: &str) -> Result<(), E> {
        self.account_text(text)?;
        if self.commit_safety_rule.is_none() {
            self.commit_safety_rule = super::sanitize::commit_safety_rule(text, None);
        }
        Ok(())
    }

    /// Accounts for a direct Base64-candidate scalar while deferring its safety result.
    fn consume_deferred_text<E: de::Error>(&mut self, text: &str) -> Result<Option<&'static str>, E> {
        self.account_text(text)?;
        Ok(super::sanitize::commit_safety_rule(text, None))
    }

    /// Enters one array or object after enforcing the explicit depth ceiling.
    fn enter_container<E: de::Error>(&mut self) -> Result<(), E> {
        self.depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| E::custom("document container-depth limit exceeded"))?;
        if self.depth > self.limits.max_depth {
            return Err(E::custom("document container-depth limit exceeded"));
        }
        self.max_depth = self.max_depth.max(self.depth);
        Ok(())
    }

    /// Leaves one successfully completed array or object.
    fn leave_container(&mut self) {
        self.depth -= 1;
    }
}

/// Seed that validates one JSON or YAML value without retaining it.
struct DocumentValueSeed<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

impl<'de> DeserializeSeed<'de> for DocumentValueSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        self.state.consume_node()?;
        deserializer.deserialize_any(DocumentValueVisitor { state: self.state })
    }
}

/// Seed that accounts for a container entry before visiting its value.
struct DocumentContainerEntrySeed<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

impl<'de> DeserializeSeed<'de> for DocumentContainerEntrySeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        self.state.consume_container_entry()?;
        DocumentValueSeed { state: self.state }.deserialize(deserializer)
    }
}

/// Seed that accounts for a YAML local tag without retaining its text.
struct DocumentTagSeed<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

impl<'de> DeserializeSeed<'de> for DocumentTagSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_identifier(DocumentTagVisitor { state: self.state })
    }
}

/// Visitor that borrows a decoded YAML local tag for resource accounting.
struct DocumentTagVisitor<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

impl<'de> Visitor<'de> for DocumentTagVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a decoded YAML local tag")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(v)
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(v)
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(&v)
    }
}

/// Seed that validates one string mapping key and returns its decoded text.
struct DocumentMappingKeySeed<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

impl<'de> DeserializeSeed<'de> for DocumentMappingKeySeed<'_> {
    type Value = String;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DocumentMappingKeyVisitor { state: self.state })
    }
}

/// Visitor that retains one decoded string key after validating tag wrappers.
struct DocumentMappingKeyVisitor<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

impl<'de> Visitor<'de> for DocumentMappingKeyVisitor<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a string mapping key, optionally wrapped in YAML local tags")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(v)?;
        Ok(v.to_owned())
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(v)?;
        Ok(v.to_owned())
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(&v)?;
        Ok(v)
    }

    fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
    where
        A: EnumAccess<'de>,
    {
        self.state.consume_node()?;
        self.state.enter_container()?;
        self.state.consume_container_entry()?;
        let ((), payload) = data.variant_seed(DocumentTagSeed { state: self.state })?;
        let key = payload.newtype_variant_seed(DocumentMappingKeySeed { state: self.state })?;
        self.state.leave_container();
        Ok(key)
    }
}

/// Recursive visitor that retains only the keys of the current mapping.
struct DocumentValueVisitor<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
}

/// Direct map value whose string meaning depends on its sibling key.
#[derive(Clone, Copy)]
enum DocumentMapValueRole {
    /// The `kind` discriminant of a potentially tagged body map.
    Kind,
    /// The `data` scalar whose Base64 representation safety may be deferred.
    Data,
}

/// Observation retained only until the surrounding map's kind is known.
#[derive(Clone, Copy)]
enum DocumentMapValueObservation {
    /// The direct value was not the Base64 discriminant or deferred string.
    Other,
    /// The direct string value selected the Base64 body variant.
    KindIsBase64,
    /// Safety result for one direct `data` string before discriminant resolution.
    DataSafety(Option<&'static str>),
}

/// Safety state retained only while one mapping's `kind` and `data` are resolved.
#[derive(Default)]
struct DocumentMapSafety {
    /// Whether the map selected the Base64 body representation.
    kind_is_base64: bool,
    /// Earliest direct `data` safety result, when it preceded global findings.
    deferred_data_safety: Option<&'static str>,
}

impl DocumentMapSafety {
    /// Applies one direct map-value observation without retaining fixture text.
    fn observe(&mut self, observation: DocumentMapValueObservation, safety_remained_unset: bool) {
        match observation {
            DocumentMapValueObservation::KindIsBase64 => self.kind_is_base64 = true,
            DocumentMapValueObservation::DataSafety(Some(rule)) if safety_remained_unset => {
                self.deferred_data_safety = Some(rule);
            },
            DocumentMapValueObservation::Other | DocumentMapValueObservation::DataSafety(_) => {},
        }
    }

    /// Restores a non-Base64 direct `data` result at its original traversal precedence.
    fn finish(self, state: &mut DocumentValidationState) {
        if !self.kind_is_base64
            && let Some(rule) = self.deferred_data_safety
        {
            state.commit_safety_rule = Some(rule);
        }
    }
}

/// Seed that observes a direct `kind` or `data` scalar while fully validating nested values.
struct DocumentMapValueSeed<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
    /// Direct scalar meaning selected by the enclosing mapping key.
    role: DocumentMapValueRole,
}

impl<'de> DeserializeSeed<'de> for DocumentMapValueSeed<'_> {
    type Value = DocumentMapValueObservation;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        self.state.consume_node()?;
        deserializer.deserialize_any(DocumentMapValueVisitor {
            state: self.state,
            role: self.role,
        })
    }
}

/// Visitor that classifies direct body-map strings without weakening nested safety scans.
struct DocumentMapValueVisitor<'a> {
    /// Shared document counters.
    state: &'a mut DocumentValidationState,
    /// Direct scalar meaning selected by the enclosing mapping key.
    role: DocumentMapValueRole,
}

impl DocumentMapValueVisitor<'_> {
    /// Accounts for one direct string and returns its map-local observation.
    fn visit_direct_text<E: de::Error>(&mut self, text: &str) -> Result<DocumentMapValueObservation, E> {
        match self.role {
            DocumentMapValueRole::Kind => {
                self.state.consume_text(text)?;
                if text == "base64" {
                    Ok(DocumentMapValueObservation::KindIsBase64)
                } else {
                    Ok(DocumentMapValueObservation::Other)
                }
            },
            DocumentMapValueRole::Data => Ok(DocumentMapValueObservation::DataSafety(
                self.state.consume_deferred_text(text)?,
            )),
        }
    }
}

impl<'de> Visitor<'de> for DocumentMapValueVisitor<'_> {
    type Value = DocumentMapValueObservation;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a direct body-map value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_str<E>(mut self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_direct_text(v)
    }

    fn visit_borrowed_str<E>(mut self, v: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_direct_text(v)
    }

    fn visit_string<E>(mut self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_direct_text(&v)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
    where
        A: EnumAccess<'de>,
    {
        self.state.enter_container()?;
        self.state.consume_container_entry()?;
        let ((), payload) = data.variant_seed(DocumentTagSeed { state: self.state })?;
        let observation = payload.newtype_variant_seed(DocumentMapValueSeed {
            state: self.state,
            role: self.role,
        })?;
        self.state.leave_container();
        Ok(observation)
    }

    fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        DocumentValueVisitor { state: self.state }.visit_seq(seq)?;
        Ok(DocumentMapValueObservation::Other)
    }

    fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        DocumentValueVisitor { state: self.state }.visit_map(map)?;
        Ok(DocumentMapValueObservation::Other)
    }
}

impl<'de> Visitor<'de> for DocumentValueVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a document value with unique mapping keys")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(v)
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(v)
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.state.consume_text(&v)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
    where
        A: EnumAccess<'de>,
    {
        self.state.enter_container()?;
        self.state.consume_container_entry()?;
        let ((), payload) = data.variant_seed(DocumentTagSeed { state: self.state })?;
        payload.newtype_variant_seed(DocumentValueSeed { state: self.state })?;
        self.state.leave_container();
        Ok(())
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        self.state.enter_container()?;
        while seq
            .next_element_seed(DocumentContainerEntrySeed { state: self.state })?
            .is_some()
        {}
        self.state.leave_container();
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        self.state.enter_container()?;
        let mut keys = HashSet::new();
        let mut map_safety = DocumentMapSafety::default();
        while let Some(key) = map.next_key_seed(DocumentMappingKeySeed { state: self.state })? {
            self.state.consume_container_entry()?;
            let role = match key.as_str() {
                "kind" => Some(DocumentMapValueRole::Kind),
                "data" => Some(DocumentMapValueRole::Data),
                _ => None,
            };
            if !keys.insert(key) {
                return Err(de::Error::custom("duplicate decoded mapping key"));
            }
            let Some(role) = role else {
                map.next_value_seed(DocumentValueSeed { state: self.state })?;
                continue;
            };
            let safety_was_unset = self.state.commit_safety_rule.is_none();
            let observation = map.next_value_seed(DocumentMapValueSeed {
                state: self.state,
                role,
            })?;
            map_safety.observe(observation, safety_was_unset && self.state.commit_safety_rule.is_none());
        }
        map_safety.finish(self.state);
        self.state.leave_container();
        Ok(())
    }
}

/// Selects JSON for `.json` paths and YAML for all other paths.
fn fixture_format(path: &Path) -> FixtureFormat {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("json") => FixtureFormat::Json,
        _ => FixtureFormat::Yaml,
    }
}

/// Returns whether a content type denotes JSON, including structured `+json` types.
pub(super) fn is_json_content_type(content_type: &str) -> bool {
    let media_type = content_type.split(';').next().map(str::trim);
    media_type.is_some_and(|value| {
        value.eq_ignore_ascii_case("application/json")
            || value
                .rsplit_once('+')
                .is_some_and(|(_, suffix)| suffix.eq_ignore_ascii_case("json"))
    })
}

/// Returns whether a content type denotes server-sent events.
fn is_sse_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("text/event-stream"))
}

/// Parses a server-sent event stream into recorded frames.
fn parse_sse_body(bytes: &[u8], limits: Option<SseParseLimits>) -> Result<RecordedBody, FixtureError> {
    let stream = std::str::from_utf8(bytes).map_err(FixtureError::SseUtf8)?;
    let mut frames = Vec::new();
    let mut builder = SseFrameBuilder::default();
    let mut done = false;

    for raw_line in stream.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            if flush_sse_frame(&mut builder, &mut frames, limits)? {
                done = true;
                break;
            }
            continue;
        }
        if line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        record_sse_field(&mut builder, field, value, frames.len(), limits)?;
    }
    if !done {
        done = flush_sse_frame(&mut builder, &mut frames, limits)?;
    }

    Ok(RecordedBody::Sse { frames, done })
}

/// Records one parsed SSE field.
#[expect(
    clippy::too_many_lines,
    reason = "the field match centralizes exact canonical-size accounting before any owned frame allocation"
)]
fn record_sse_field<'a>(
    builder: &mut SseFrameBuilder<'a>,
    field: &str,
    value: &'a str,
    completed_frames: usize,
    limits: Option<SseParseLimits>,
) -> Result<(), FixtureError> {
    match field {
        "event" => {
            let (field_bytes, event_bytes) = prospective_sse_singleton_size(
                builder.field_bytes,
                builder.event_bytes,
                b"event: \n".len(),
                value.len(),
            )?;
            validate_sse_frame_size(field_bytes, !builder.data_lines.is_empty(), limits)?;
            builder.field_bytes = field_bytes;
            builder.event_bytes = event_bytes;
            builder.event = Some(value);
        },
        "data" => {
            let ordinary = !builder.data_lines.is_empty() || value != "[DONE]";
            if ordinary && limits.is_some_and(|limit| completed_frames >= limit.max_frames) {
                return Err(sse_parse_error("recorded SSE frame count exceeded replay limit"));
            }
            let field_bytes = builder
                .field_bytes
                .checked_add(b"data: \n".len())
                .and_then(|size| size.checked_add(value.len()))
                .ok_or_else(|| sse_parse_error("recorded SSE frame size overflowed"))?;
            validate_sse_frame_size(field_bytes, true, limits)?;
            builder.field_bytes = field_bytes;
            builder.data_lines.push(value);
        },
        "id" => {
            let (field_bytes, id_bytes) =
                prospective_sse_singleton_size(builder.field_bytes, builder.id_bytes, b"id: \n".len(), value.len())?;
            validate_sse_frame_size(field_bytes, !builder.data_lines.is_empty(), limits)?;
            builder.field_bytes = field_bytes;
            builder.id_bytes = id_bytes;
            builder.id = Some(value);
        },
        "retry" => {
            let retry = value.parse().ok();
            let value_len = retry.map_or(0, decimal_len);
            let (field_bytes, retry_bytes) = if retry.is_some() {
                prospective_sse_singleton_size(builder.field_bytes, builder.retry_bytes, b"retry: \n".len(), value_len)?
            } else {
                (
                    builder
                        .field_bytes
                        .checked_sub(builder.retry_bytes)
                        .ok_or_else(|| sse_parse_error("recorded SSE frame size overflowed"))?,
                    0,
                )
            };
            validate_sse_frame_size(field_bytes, !builder.data_lines.is_empty(), limits)?;
            builder.field_bytes = field_bytes;
            builder.retry_bytes = retry_bytes;
            builder.retry = retry;
        },
        _ => {},
    }
    Ok(())
}

/// Flushes a block and reports an exact full-event `[DONE]` marker.
fn flush_sse_frame(
    builder: &mut SseFrameBuilder<'_>,
    frames: &mut Vec<SseFrame>,
    limits: Option<SseParseLimits>,
) -> Result<bool, FixtureError> {
    let done = builder.data_lines.len() == 1 && builder.data_lines.first().is_some_and(|data| *data == "[DONE]");
    if done {
        *builder = SseFrameBuilder::default();
        return Ok(true);
    }
    if !builder.data_lines.is_empty() {
        if limits.is_some_and(|limit| frames.len() >= limit.max_frames) {
            return Err(sse_parse_error("recorded SSE frame count exceeded replay limit"));
        }
        frames.push(SseFrame {
            event: builder.event.take().map(str::to_owned),
            data: std::mem::take(&mut builder.data_lines).join("\n"),
            id: builder.id.take().map(str::to_owned),
            retry: builder.retry.take(),
        });
    }
    *builder = SseFrameBuilder::default();
    Ok(false)
}

/// Computes a replacement singleton field and total without mutating the builder.
fn prospective_sse_singleton_size(
    current: usize,
    previous: usize,
    overhead: usize,
    value: usize,
) -> Result<(usize, usize), FixtureError> {
    let singleton = overhead
        .checked_add(value)
        .ok_or_else(|| sse_parse_error("recorded SSE frame size overflowed"))?;
    let total = current
        .checked_sub(previous)
        .and_then(|size| size.checked_add(singleton))
        .ok_or_else(|| sse_parse_error("recorded SSE frame size overflowed"))?;
    Ok((total, singleton))
}

/// Checks the canonical frame total, including its final blank-line newline.
fn validate_sse_frame_size(
    field_bytes: usize,
    has_data: bool,
    limits: Option<SseParseLimits>,
) -> Result<(), FixtureError> {
    if !has_data {
        return Ok(());
    }
    let frame_bytes = field_bytes
        .checked_add(1)
        .ok_or_else(|| sse_parse_error("recorded SSE frame size overflowed"))?;
    if limits.is_some_and(|limit| frame_bytes > limit.max_frame_bytes) {
        return Err(sse_parse_error("recorded SSE frame exceeded replay limit"));
    }
    Ok(())
}

/// Counts decimal digits without allocating a temporary retry string.
fn decimal_len(mut value: u64) -> usize {
    let mut digits = 1_usize;
    while value >= 10 {
        value /= 10;
        digits = digits.saturating_add(1);
    }
    digits
}

/// Creates one opaque allocation-bound parser error.
fn sse_parse_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

/// Renders recorded SSE frames with canonical LF line endings.
fn render_sse_body(frames: &[SseFrame], done: bool) -> Vec<u8> {
    let mut rendered = Vec::new();
    for frame in frames {
        if let Some(event) = &frame.event {
            push_sse_field(&mut rendered, "event", event);
        }
        for data_line in frame.data.split('\n') {
            push_sse_field(&mut rendered, "data", data_line);
        }
        if let Some(id) = &frame.id {
            push_sse_field(&mut rendered, "id", id);
        }
        if let Some(retry) = frame.retry {
            push_sse_field(&mut rendered, "retry", &retry.to_string());
        }
        rendered.push(b'\n');
    }
    if done {
        rendered.extend_from_slice(b"data: [DONE]\n\n");
    }
    rendered
}

/// Appends one canonical SSE field line to a byte buffer.
fn push_sse_field(output: &mut Vec<u8>, name: &str, value: &str) {
    output.extend_from_slice(name.as_bytes());
    output.extend_from_slice(b": ");
    output.extend_from_slice(value.as_bytes());
    output.push(b'\n');
}

/// Replaces only exact `${MODEL}` string values in a JSON tree.
fn bind_model_value(value: &mut Value, model: &str) {
    match value {
        Value::String(current) if current == "${MODEL}" => model.clone_into(current),
        Value::Array(values) => {
            for element in values {
                bind_model_value(element, model);
            }
        },
        Value::Object(values) => {
            for nested_value in values.values_mut() {
                bind_model_value(nested_value, model);
            }
        },
        _ => {},
    }
}

/// Returns whether a JSON tree contains an exact string value.
fn contains_exact_string(value: &Value, expected: &str) -> bool {
    match value {
        Value::String(current) => current == expected,
        Value::Array(values) => values.iter().any(|value| contains_exact_string(value, expected)),
        Value::Object(values) => values.values().any(|value| contains_exact_string(value, expected)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// Replaces exact string values in a JSON tree without rewriting substrings.
fn bind_exact_string(value: &mut Value, expected: &str, replacement: &str) {
    match value {
        Value::String(current) if current == expected => replacement.clone_into(current),
        Value::Array(values) => {
            for value in values {
                bind_exact_string(value, expected, replacement);
            }
        },
        Value::Object(values) => {
            for value in values.values_mut() {
                bind_exact_string(value, expected, replacement);
            }
        },
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {},
    }
}


#[cfg(test)]
mod tests {
    //! Behavioral tests for the versioned inference fixture schema.

    use std::{fs, io::Cursor};

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde_json::{Value, json};

    use super::*;

    /// A complete, valid version-one fixture expressed as YAML.
    const COMPLETE_FIXTURE: &str = r#"
version: 1
scenario_id: responses-basic
protocol: openai_responses
provenance:
  kind: live
  provider: openai
  model: gpt-5
  source_id: capture-123
normalization:
  version: 1
  linked_ids:
    resp_1: response-1
turns:
  - name: create-response
    client:
      request:
        method: POST
        path: /v1/responses
        headers:
          content-type:
            - application/json
        body:
          kind: json
          value:
            model: model-under-test
            input: hello
      response:
        status: 200
        headers: {}
        body:
          kind: json
          value:
            id: resp_1
    upstream:
      request:
        method: POST
        path: /v1/responses
        headers: {}
        body:
          kind: json
          value:
            model: model-under-test
      response:
        status: 200
        headers:
          x-request-id:
            - request-1
        body:
          kind: sse
          frames:
            - event: response.created
              data: '{"type":"response.created"}'
              id: "1"
              retry: 1000
          done: false
"#;

    /// A complete, valid version-one scenario expressed as YAML.
    const COMPLETE_SCENARIO: &str = "
version: 1
id: responses-basic
description: Exercise a basic Responses request.
protocol: openai_responses
example_config: examples/configs/ai/responses.yaml
upstream_authority: upstream.test
features:
  - responses
turns:
  - name: create-response
    request:
      method: POST
      path: /v1/responses
      headers: {}
      body:
        kind: json
        value:
          model: ${MODEL}
          nested:
            models:
              - ${MODEL}
              - prefix-${MODEL}
    expect:
      client_status: 200
      client_body_kind: json
      upstream_path: /v1/responses
      upstream_body_kind: json
";

    #[test]
    fn wire_fixture_loads_a_complete_version_one_yaml_document() {
        // Arrange
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(temporary_file.path(), COMPLETE_FIXTURE).unwrap();

        // Act
        let fixture = WireFixture::load(temporary_file.path()).unwrap();

        // Assert
        assert_eq!(fixture.version, WIRE_FIXTURE_VERSION);
        assert_eq!(fixture.scenario_id, "responses-basic");
        assert_eq!(fixture.protocol, InferenceProtocol::OpenaiResponses);
        assert_eq!(fixture.turns.len(), 1);
        assert_eq!(fixture.turns[0].upstream.response.status, 200);
        assert!(matches!(
            &fixture.turns[0].upstream.response.body,
            RecordedBody::Sse {
                done: false,
                frames
            } if frames[0].event.as_deref() == Some("response.created")
        ));
    }

    #[test]
    fn wire_fixture_json_rejects_decoded_duplicate_keys_at_payload_depth() {
        let fixture: WireFixture = serde_yaml::from_str(COMPLETE_FIXTURE).unwrap();
        let mut document = serde_json::to_string_pretty(&fixture).unwrap();
        let needle = "\"input\": \"hello\"";
        let offset = document.find(needle).expect("payload mutation anchor should exist");
        document.replace_range(
            offset..offset + needle.len(),
            "\"input\": \"first\", \"inp\\u0075t\": \"second\"",
        );
        let temporary_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        fs::write(temporary_file.path(), document).unwrap();

        let result = WireFixture::load(temporary_file.path());

        assert!(matches!(result, Err(FixtureError::JsonFixture { .. })));
    }

    #[test]
    fn wire_fixture_json_rejects_trailing_documents() {
        let fixture: WireFixture = serde_yaml::from_str(COMPLETE_FIXTURE).unwrap();
        let mut document = serde_json::to_vec(&fixture).unwrap();
        document.extend_from_slice(b"\n{}");
        let temporary_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        fs::write(temporary_file.path(), document).unwrap();

        assert!(matches!(
            WireFixture::load(temporary_file.path()),
            Err(FixtureError::JsonFixture { .. })
        ));
    }

    #[test]
    fn every_recorded_body_variant_rejects_unknown_fields() {
        for (index, document) in [
            r#"{"kind":"empty","unknown":null}"#,
            r#"{"kind":"json","value":{},"unknown":null}"#,
            r#"{"kind":"sse","frames":[],"done":false,"unknown":null}"#,
            r#"{"kind":"base64","data":"","unknown":null}"#,
        ]
        .into_iter()
        .enumerate()
        {
            assert!(
                serde_json::from_str::<RecordedBody>(document).is_err(),
                "variant {index}"
            );
        }
    }

    #[test]
    fn fixture_document_ceiling_has_one_full_encoded_turn_of_headroom() {
        assert_eq!(MAX_ONE_TURN_WIRE_BODY_BYTES, 48 * 1024 * 1024);
        assert_eq!(MAX_BASE64_ONE_TURN_BODY_BYTES, 67_108_872);
        assert_eq!(MAX_FIXTURE_DOCUMENT_BYTES, 134_217_744);
        assert!(!is_fixture_document_oversized(
            u64::try_from(MAX_FIXTURE_DOCUMENT_BYTES).unwrap()
        ));
        assert!(is_fixture_document_oversized(
            u64::try_from(MAX_FIXTURE_DOCUMENT_BYTES).unwrap() + 1
        ));

        let temporary_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        temporary_file
            .as_file()
            .set_len(u64::try_from(MAX_FIXTURE_DOCUMENT_BYTES).unwrap() + 1)
            .unwrap();
        assert!(matches!(
            WireFixture::load(temporary_file.path()),
            Err(FixtureError::PersistedDocumentTooLarge { .. })
        ));
    }

    #[test]
    fn exact_sized_reader_rejects_short_and_growing_inputs_without_spare_capacity() {
        let document = read_exact_sized_document(Cursor::new(b"1234"), 4).unwrap();
        assert_eq!(document.as_slice(), b"1234");

        assert!(matches!(
            read_exact_sized_document(Cursor::new(b"123"), 4),
            Err(ExactDocumentReadError::LengthChanged)
        ));
        assert!(matches!(
            read_exact_sized_document(Cursor::new(b"12345"), 4),
            Err(ExactDocumentReadError::LengthChanged)
        ));
    }

    #[test]
    fn exact_sized_reader_reports_injected_allocation_failure_without_reading() {
        struct PanicOnRead;
        impl Read for PanicOnRead {
            fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
                panic!("reader must not be touched after allocation failure");
            }
        }

        let error = read_exact_sized_document_with_allocator(PanicOnRead, 4, |_len| Err(()))
            .expect_err("allocation failure must be recoverable");

        assert!(matches!(error, ExactDocumentReadError::Allocation));
    }

    #[test]
    fn fixture_reader_applies_a_small_test_ceiling_before_allocation() {
        let temporary_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        fs::write(temporary_file.path(), b"12345").unwrap();

        let limits = |max_encoded_bytes| PersistedDocumentLimits {
            max_encoded_bytes,
            validation: DocumentValidationLimits::TEST_PERMISSIVE,
            kind: PersistedDocumentKind::WireFixture,
        };

        assert!(matches!(
            read_fixture_file(temporary_file.path(), limits(4)),
            Err(FixtureError::PersistedDocumentTooLarge { .. })
        ));
        assert_eq!(
            read_fixture_file(temporary_file.path(), limits(5)).unwrap().as_slice(),
            b"12345"
        );
    }

    #[test]
    fn fixture_reader_maps_injected_allocation_failure_to_an_opaque_fixture_error() {
        let temporary_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        fs::write(temporary_file.path(), b"1234").unwrap();
        let limits = PersistedDocumentLimits {
            max_encoded_bytes: 4,
            validation: DocumentValidationLimits::TEST_PERMISSIVE,
            kind: PersistedDocumentKind::WireFixture,
        };

        let error = read_fixture_file_with_limits_and_allocator(temporary_file.path(), limits, |_len| Err(()))
            .expect_err("allocation failure must not panic");

        assert!(matches!(error, FixtureError::PersistedDocumentAllocation));
        assert_eq!(error.to_string(), "persisted fixture document allocation failed");
    }

    #[test]
    fn streaming_json_validation_enforces_node_budget_before_typed_deserialization() {
        assert_json_budget(
            b"[0,1]",
            DocumentValidationLimits {
                max_nodes: 2,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document node limit exceeded",
        );
    }

    #[test]
    fn streaming_json_validation_enforces_entry_budget_before_typed_deserialization() {
        assert_json_budget(
            b"[0,1]",
            DocumentValidationLimits {
                max_container_entries: 1,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document container-entry limit exceeded",
        );
    }

    #[test]
    fn streaming_json_validation_enforces_string_budget_before_typed_deserialization() {
        assert_json_budget(
            br#"{"key":"value"}"#,
            DocumentValidationLimits {
                max_decoded_string_bytes: 7,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document decoded-string limit exceeded",
        );
    }

    #[test]
    fn streaming_json_validation_enforces_depth_budget_before_typed_deserialization() {
        assert_json_budget(
            b"[[0]]",
            DocumentValidationLimits {
                max_depth: 1,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document container-depth limit exceeded",
        );
    }

    fn assert_json_budget(document: &[u8], limits: DocumentValidationLimits, expected_rule: &str) {
        let error = validate_json_document_with_limits(document, limits).unwrap_err();
        assert!(error.to_string().contains(expected_rule), "{error}");
    }

    #[test]
    fn streaming_json_validation_scans_decoded_unknown_keys_and_values() {
        for (document, expected_rule) in [
            (br#"{"Beare\u0072 unsafe-key":"safe"}"#.as_slice(), "bearer token"),
            (
                br#"{"unknown":"/Users/private/value"}"#.as_slice(),
                "absolute user path",
            ),
            (
                br#"{"unknown":"prefix=/home/private/value"}"#.as_slice(),
                "absolute home path",
            ),
            (
                br#"{"unknown":"prefix C:\\private\\value"}"#.as_slice(),
                "Windows drive path",
            ),
            (br#"{"unknown":"prefix BeArEr\tcredential"}"#.as_slice(), "bearer token"),
        ] {
            let validation =
                validate_json_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).unwrap();
            assert_eq!(validation.commit_safety_rule, Some(expected_rule));
        }

        let safe = validate_json_document_with_limits(
            br#"{"unknown":"https://example.test/home/private/value"}"#,
            DocumentValidationLimits::TEST_PERMISSIVE,
        )
        .unwrap();
        assert_eq!(safe.commit_safety_rule, None);
    }

    #[test]
    fn streaming_yaml_validation_uses_boundary_aware_safety_for_unknown_fields() {
        for (document, expected_rule) in [
            (
                b"unknown: \"prefix=/Users/private/value\"\n".as_slice(),
                "absolute user path",
            ),
            (b"unknown: \"prefix BeArEr\\tcredential\"\n".as_slice(), "bearer token"),
            (
                b"unknown: \"file:///home/private/value\"\n".as_slice(),
                "absolute home path",
            ),
        ] {
            let validation =
                validate_yaml_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).unwrap();
            assert_eq!(validation.commit_safety_rule, Some(expected_rule));
        }

        let safe = validate_yaml_document_with_limits(
            b"unknown: \"https://example.test/home/private/value\"\n",
            DocumentValidationLimits::TEST_PERMISSIVE,
        )
        .unwrap();
        assert_eq!(safe.commit_safety_rule, None);
    }

    #[test]
    fn streaming_json_validation_suppresses_only_direct_base64_data_safety_regardless_of_key_order() {
        for document in [
            br#"{"kind":"base64","data":"/home/AA"}"#.as_slice(),
            br#"{"data":"/Users/A","kind":"base64"}"#.as_slice(),
        ] {
            let validation =
                validate_json_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).unwrap();
            assert_eq!(validation.commit_safety_rule, None);
        }

        for document in [
            br#"{"kind":"json","data":"/home/private"}"#.as_slice(),
            br#"{"data":"/Users/private","kind":"sse"}"#.as_slice(),
            br#"{"kind":"base64","data":{"nested":"/home/private"}}"#.as_slice(),
        ] {
            let validation =
                validate_json_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).unwrap();
            assert!(validation.commit_safety_rule.is_some());
        }
    }

    #[test]
    fn streaming_yaml_validation_suppresses_only_direct_base64_data_safety_regardless_of_key_order() {
        for document in [
            b"kind: base64\ndata: /home/AA\n".as_slice(),
            b"data: /Users/A\nkind: base64\n".as_slice(),
        ] {
            let validation =
                validate_yaml_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).unwrap();
            assert_eq!(validation.commit_safety_rule, None);
        }

        for document in [
            b"kind: json\ndata: /home/private\n".as_slice(),
            b"data: /Users/private\nkind: sse\n".as_slice(),
            b"kind: base64\ndata:\n  nested: /home/private\n".as_slice(),
        ] {
            let validation =
                validate_yaml_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).unwrap();
            assert!(validation.commit_safety_rule.is_some());
        }
    }

    #[test]
    fn decoded_base64_safety_remains_authoritative_after_raw_representation_suppression() {
        let mut fixture: WireFixture = serde_yaml::from_str(COMPLETE_FIXTURE).unwrap();

        for (suffix, encoded) in [(".json", "/home/AA"), (".yaml", "/Users/A")] {
            fixture.turns[0].client.request.body = RecordedBody::Base64 {
                data: encoded.to_owned(),
            };
            let temporary_file = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
            fixture.write(temporary_file.path()).unwrap();

            let loaded = WireFixture::load_for_commit_check(temporary_file.path()).unwrap();
            assert!(loaded.raw_commit_safety_error.is_none());
            crate::inference_fixture::validate_commit_safe(&loaded.fixture).unwrap();
        }

        fixture.turns[0].client.request.body = RecordedBody::Base64 {
            data: STANDARD.encode(b"prefix /home/private/value"),
        };

        for suffix in [".json", ".yaml"] {
            let temporary_file = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
            fixture.write(temporary_file.path()).unwrap();

            let loaded = WireFixture::load_for_commit_check(temporary_file.path()).unwrap();
            assert!(loaded.raw_commit_safety_error.is_none());
            let error = crate::inference_fixture::validate_commit_safe(&loaded.fixture).unwrap_err();
            assert!(matches!(
                error,
                FixtureError::CommitSafety {
                    rule: "absolute home path",
                    ..
                }
            ));
        }
    }

    #[test]
    fn streaming_yaml_validation_rejects_duplicate_keys_and_multiple_documents() {
        for document in [
            b"outer:\n  input: first\n  \"inp\\u0075t\": second\n".as_slice(),
            b"!first input: first\ninput: second\n".as_slice(),
            b"!first input: first\n!second input: second\n".as_slice(),
            b"version: 1\n---\nversion: 1\n".as_slice(),
        ] {
            assert!(validate_yaml_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).is_err());
        }
    }

    #[test]
    fn streaming_yaml_validation_accounts_for_local_tags_on_mapping_keys() {
        let ordinary = b"field: value";
        validate_yaml_document_with_limits(
            ordinary,
            DocumentValidationLimits {
                max_nodes: 2,
                max_container_entries: 1,
                max_decoded_string_bytes: 10,
                max_depth: 1,
            },
        )
        .unwrap();

        let tagged = b"!tag field: value";
        assert_yaml_budget_with(tagged, |limits| limits.max_nodes = 2, "document node limit exceeded");
        assert_yaml_budget_with(
            tagged,
            |limits| limits.max_container_entries = 1,
            "document container-entry limit exceeded",
        );
        assert_yaml_budget_with(
            tagged,
            |limits| limits.max_decoded_string_bytes = 10,
            "document decoded-string limit exceeded",
        );
        assert_yaml_budget_with(
            tagged,
            |limits| limits.max_depth = 1,
            "document container-depth limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_counts_decoded_key_tag_text_exactly() {
        let document = b"!key%20tag field: value";
        validate_yaml_document_with_limits(
            document,
            DocumentValidationLimits {
                max_decoded_string_bytes: 17,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
        )
        .unwrap();
        assert_yaml_budget(
            document,
            DocumentValidationLimits {
                max_decoded_string_bytes: 16,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document decoded-string limit exceeded",
        );

        let safety = validate_yaml_document_with_limits(
            b"!Bearer%20placeholder field: value",
            DocumentValidationLimits::TEST_PERMISSIVE,
        )
        .unwrap();
        assert_eq!(safety.commit_safety_rule, Some("bearer token"));
    }

    #[test]
    fn streaming_yaml_validation_recursively_accounts_for_tagged_key_aliases() {
        let document = b"&field !tag shared: first\n*field: second\n";
        assert!(serde_yaml::from_slice::<BTreeMap<String, String>>(document).is_ok());
        assert!(
            validate_yaml_document_with_limits(document, DocumentValidationLimits::TEST_PERMISSIVE).is_err(),
            "expanded tagged alias must collide with its decoded key"
        );

        let single_use = b"&field !tag shared: first\nother: second\n";
        assert_yaml_budget(
            single_use,
            DocumentValidationLimits {
                max_depth: 1,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document container-depth limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_counts_tag_text_when_an_alias_is_used_as_a_key() {
        let value_anchor_used_as_a_key = b"source: &field !tag shared\n*field: second\n";
        validate_yaml_document_with_limits(
            value_anchor_used_as_a_key,
            DocumentValidationLimits {
                max_decoded_string_bytes: 30,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
        )
        .unwrap();
        assert_yaml_budget(
            value_anchor_used_as_a_key,
            DocumentValidationLimits {
                max_decoded_string_bytes: 29,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document decoded-string limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_rejects_non_string_mapping_keys_like_typed_schemas() {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct StringKeySchema {
            /// Only field accepted by this representative persisted schema.
            #[expect(dead_code, reason = "deserialization behavior is the assertion")]
            field: String,
        }

        for document in ["1: value\n", "[key]: value\n", "{nested: key}: value\n"] {
            assert!(serde_yaml::from_str::<StringKeySchema>(document).is_err());
            assert!(
                validate_yaml_document_with_limits(document.as_bytes(), DocumentValidationLimits::TEST_PERMISSIVE,)
                    .is_err(),
                "{document}"
            );
        }
    }

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    enum TaggedYamlForm {
        Unit,
        Scalar(String),
        Sequence(Vec<String>),
        Mapping { nested: Vec<String> },
    }

    #[test]
    fn bounded_yaml_parser_matches_typed_deserialization_for_local_tag_forms() {
        let path = Path::new("tagged.yaml");
        for document in [
            "!unit\n",
            "!scalar\n",
            "!scalar value\n",
            "!sequence [one, two]\n",
            "!mapping {nested: [one, two]}\n",
        ] {
            let expected: TaggedYamlForm = serde_yaml::from_str(document).unwrap();

            let parsed: ParsedDocument<TaggedYamlForm> = parse_strict_yaml_document_with_limits(
                document.as_bytes(),
                path,
                DocumentValidationLimits::TEST_PERMISSIVE,
            )
            .unwrap();

            assert_eq!(parsed.value, expected, "{document}");
        }
    }

    #[test]
    fn bounded_yaml_parser_rejects_invalid_local_tag_variant_payloads() {
        let path = Path::new("tagged.yaml");
        for document in [
            "!unit unexpected\n",
            "!scalar [unexpected]\n",
            "!sequence unexpected\n",
            "!mapping unexpected\n",
            "!unknown value\n",
        ] {
            assert!(serde_yaml::from_str::<TaggedYamlForm>(document).is_err(), "{document}");
            assert!(
                parse_strict_yaml_document_with_limits::<TaggedYamlForm>(
                    document.as_bytes(),
                    path,
                    DocumentValidationLimits::TEST_PERMISSIVE,
                )
                .is_err(),
                "{document}"
            );
        }
    }

    #[test]
    fn streaming_yaml_validation_accounts_for_tag_wrappers_and_payloads() {
        let document = b"!mapping {nested: !sequence [one]}";
        assert_yaml_budget_with(document, |limits| limits.max_nodes = 3, "document node limit exceeded");
        assert_yaml_budget_with(
            document,
            |limits| limits.max_container_entries = 2,
            "document container-entry limit exceeded",
        );
        assert_yaml_budget_with(
            document,
            |limits| limits.max_decoded_string_bytes = 10,
            "document decoded-string limit exceeded",
        );
        assert_yaml_budget_with(
            document,
            |limits| limits.max_depth = 2,
            "document container-depth limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_rejects_duplicate_keys_inside_a_tagged_map() {
        assert!(
            validate_yaml_document_with_limits(
                b"!mapping {input: first, input: second}",
                DocumentValidationLimits::TEST_PERMISSIVE,
            )
            .is_err()
        );
    }

    #[test]
    fn streaming_yaml_validation_counts_each_tagged_alias_expansion() {
        let document = b"base: &payload !sequence [one]\ncopy: *payload\n";
        validate_yaml_document_with_limits(
            document,
            DocumentValidationLimits {
                max_nodes: 7,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
        )
        .unwrap();

        assert_yaml_budget(
            document,
            DocumentValidationLimits {
                max_nodes: 6,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document node limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_enforces_node_budget() {
        assert_yaml_budget(
            b"values: [one, two]",
            DocumentValidationLimits {
                max_nodes: 3,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document node limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_enforces_entry_budget() {
        assert_yaml_budget(
            b"values: [one, two]",
            DocumentValidationLimits {
                max_container_entries: 2,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document container-entry limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_enforces_depth_budget() {
        assert_yaml_budget(
            b"outer:\n  inner:\n    value: true\n",
            DocumentValidationLimits {
                max_depth: 1,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document container-depth limit exceeded",
        );
    }

    #[test]
    fn streaming_yaml_validation_enforces_decoded_string_budget() {
        assert_yaml_budget(
            b"key: decoded-value\n",
            DocumentValidationLimits {
                max_decoded_string_bytes: 10,
                ..DocumentValidationLimits::TEST_PERMISSIVE
            },
            "document decoded-string limit exceeded",
        );
    }

    fn assert_yaml_budget(document: &[u8], limits: DocumentValidationLimits, expected_rule: &str) {
        let error = validate_yaml_document_with_limits(document, limits).unwrap_err();
        assert!(error.to_string().contains(expected_rule), "{error}");
    }

    fn assert_yaml_budget_with(
        document: &[u8],
        update: impl FnOnce(&mut DocumentValidationLimits),
        expected_rule: &str,
    ) {
        let mut limits = DocumentValidationLimits::TEST_PERMISSIVE;
        update(&mut limits);
        assert_yaml_budget(document, limits, expected_rule);
    }

    #[test]
    fn scenario_loader_applies_a_configurable_document_limit() {
        let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        fs::write(temporary_file.path(), COMPLETE_SCENARIO).unwrap();

        assert!(matches!(
            load_scenario_with_limits(
                temporary_file.path(),
                PersistedDocumentLimits {
                    max_encoded_bytes: 16,
                    validation: DocumentValidationLimits::TEST_PERMISSIVE,
                    kind: PersistedDocumentKind::Scenario,
                },
                usize::MAX,
            ),
            Err(FixtureError::PersistedDocumentTooLarge { .. })
        ));
    }

    #[test]
    fn scenario_loader_applies_a_configurable_request_body_limit() {
        let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        fs::write(temporary_file.path(), COMPLETE_SCENARIO).unwrap();

        let body_limited = load_scenario_with_limits(
            temporary_file.path(),
            PersistedDocumentLimits {
                max_encoded_bytes: COMPLETE_SCENARIO.len() + 1,
                validation: DocumentValidationLimits {
                    max_nodes: 1_000,
                    max_container_entries: 1_000,
                    max_decoded_string_bytes: COMPLETE_SCENARIO.len() * 2,
                    max_depth: 32,
                },
                kind: PersistedDocumentKind::Scenario,
            },
            8,
        )
        .expect_err("typed scenario request bodies must be bounded during load");
        assert_eq!(body_limited.to_string(), "scenario request body exceeded replay limit");
    }

    #[test]
    fn scenario_public_loader_rejects_its_dedicated_ceiling_before_allocation() {
        let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        temporary_file
            .as_file()
            .set_len(u64::try_from(MAX_SCENARIO_DOCUMENT_BYTES).unwrap() + 1)
            .unwrap();

        assert!(matches!(
            InferenceScenario::load(temporary_file.path()),
            Err(FixtureError::PersistedDocumentTooLarge { kind: "scenario" })
        ));
    }

    #[test]
    fn scenario_loader_rejects_duplicate_keys_and_multiple_yaml_documents() {
        for document in [
            COMPLETE_SCENARIO.replacen("version: 1", "version: 1\nversion: 1", 1),
            format!("{COMPLETE_SCENARIO}\n---\n{COMPLETE_SCENARIO}"),
        ] {
            let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
            fs::write(temporary_file.path(), document).unwrap();

            assert!(matches!(
                InferenceScenario::load(temporary_file.path()),
                Err(FixtureError::YamlFixture { .. })
            ));
        }
    }

    #[test]
    fn scenario_loader_matches_typed_yaml_for_local_tagged_enum_fields() {
        let document = COMPLETE_SCENARIO
            .replacen("protocol: openai_responses", "protocol: !openai_responses", 1)
            .replacen("client_body_kind: json", "client_body_kind: !json", 1)
            .replacen("upstream_body_kind: json", "upstream_body_kind: !json", 1);
        let expected: InferenceScenario = serde_yaml::from_str(&document).unwrap();
        let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        fs::write(temporary_file.path(), document).unwrap();

        let loaded = InferenceScenario::load(temporary_file.path()).unwrap();

        assert_eq!(loaded, expected);
        assert_eq!(loaded.protocol, InferenceProtocol::OpenaiResponses);
        assert_eq!(loaded.turns[0].expect.client_body_kind, BodyKind::Json);
        assert_eq!(loaded.turns[0].expect.upstream_body_kind, BodyKind::Json);
    }

    #[test]
    fn scenario_public_loader_matches_typed_yaml_for_local_tagged_mapping_keys() {
        let document = COMPLETE_SCENARIO
            .replacen("version: 1", "!schema version: 1", 1)
            .replacen("protocol: openai_responses", "!wire protocol: openai_responses", 1)
            .replacen("    request:", "    !turn request:", 1);
        let expected: InferenceScenario = serde_yaml::from_str(&document).unwrap();
        let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        fs::write(temporary_file.path(), document).unwrap();

        let loaded = InferenceScenario::load(temporary_file.path()).unwrap();

        assert_eq!(loaded, expected);
    }

    #[test]
    fn scenario_loader_applies_small_budgets_to_local_tagged_mapping_keys() {
        let nested_body = "        value:\n          model: ${MODEL}\n          nested:\n            models:\n              - ${MODEL}\n              - prefix-${MODEL}\n";
        let ordinary = COMPLETE_SCENARIO.replace(nested_body, "        value:\n          model: ${MODEL}\n");
        let tagged = ordinary.replacen("          model: ${MODEL}", "          !payload model: ${MODEL}", 1);
        let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();

        let limits = |max_encoded_bytes| PersistedDocumentLimits {
            max_encoded_bytes,
            validation: DocumentValidationLimits {
                max_nodes: 100,
                max_container_entries: 100,
                max_decoded_string_bytes: 1_000,
                max_depth: 6,
            },
            kind: PersistedDocumentKind::Scenario,
        };
        fs::write(temporary_file.path(), &ordinary).unwrap();
        load_scenario_with_limits(temporary_file.path(), limits(ordinary.len()), MAX_REQUEST_BODY_BYTES).unwrap();

        fs::write(temporary_file.path(), &tagged).unwrap();
        let error = load_scenario_with_limits(temporary_file.path(), limits(tagged.len()), MAX_REQUEST_BODY_BYTES)
            .expect_err("a tagged key wrapper must consume depth");
        assert!(error.to_string().contains("failed to parse YAML fixture"));
    }

    #[test]
    fn scenario_public_loader_rejects_duplicate_decoded_tagged_mapping_keys() {
        for duplicate in [
            "!schema version: 1\nversion: 1",
            "!first version: 1\n!second version: 1",
        ] {
            let document = COMPLETE_SCENARIO.replacen("version: 1", duplicate, 1);
            let temporary_file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
            fs::write(temporary_file.path(), document).unwrap();

            assert!(matches!(
                InferenceScenario::load(temporary_file.path()),
                Err(FixtureError::YamlFixture { .. })
            ));
        }
    }

    #[test]
    fn wire_fixture_load_rejects_an_unsupported_version() {
        // Arrange
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(
            temporary_file.path(),
            COMPLETE_FIXTURE.replacen("version: 1", "version: 2", 1),
        )
        .unwrap();

        // Act
        let result = WireFixture::load(temporary_file.path());

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn wire_fixture_rejects_an_unsupported_normalization_version() {
        // Catches validating only the outer wire-fixture schema version.
        let source_file = tempfile::NamedTempFile::new().unwrap();
        let output_file = tempfile::NamedTempFile::new().unwrap();
        let document = COMPLETE_FIXTURE.replacen("normalization:\n  version: 1", "normalization:\n  version: 99", 1);
        fs::write(source_file.path(), document).unwrap();

        assert!(matches!(
            WireFixture::load(source_file.path()),
            Err(FixtureError::UnsupportedNormalizationVersion { version: 99 })
        ));

        let mut fixture: WireFixture = serde_yaml::from_str(COMPLETE_FIXTURE).unwrap();
        fixture.normalization.version = 99;
        assert!(matches!(
            fixture.write(output_file.path()),
            Err(FixtureError::UnsupportedNormalizationVersion { version: 99 })
        ));
    }

    #[test]
    fn wire_fixture_load_rejects_unknown_fields() {
        // Arrange
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(temporary_file.path(), format!("{COMPLETE_FIXTURE}unexpected: value\n")).unwrap();

        // Act
        let result = WireFixture::load(temporary_file.path());

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn wire_fixture_write_then_load_preserves_the_fixture() {
        // Arrange
        let source_file = tempfile::NamedTempFile::new().unwrap();
        let output_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(source_file.path(), COMPLETE_FIXTURE).unwrap();
        let expected = WireFixture::load(source_file.path()).unwrap();

        // Act
        expected.write(output_file.path()).unwrap();
        let loaded = WireFixture::load(output_file.path()).unwrap();

        // Assert
        assert_eq!(loaded, expected);
    }

    #[test]
    fn wire_fixture_json_serialization_enforces_the_loader_document_ceiling() {
        let source_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(source_file.path(), COMPLETE_FIXTURE).unwrap();
        let fixture = WireFixture::load(source_file.path()).unwrap();
        let diagnostic_path = Path::new("bounded-fixture.json");
        let document = serialize_pretty_json_with_limit(&fixture, diagnostic_path, usize::MAX).unwrap();

        assert_eq!(document.last(), Some(&b'\n'));
        assert_eq!(
            serialize_pretty_json_with_limit(&fixture, diagnostic_path, document.len()).unwrap(),
            document
        );
        assert!(matches!(
            serialize_pretty_json_with_limit(&fixture, diagnostic_path, document.len() - 1),
            Err(FixtureError::PersistedDocumentTooLarge { kind: "wire fixture" })
        ));
        assert!(matches!(
            serialize_pretty_json_with_limits(
                &fixture,
                diagnostic_path,
                usize::MAX,
                DocumentValidationLimits {
                    max_nodes: 1,
                    ..DocumentValidationLimits::TEST_PERMISSIVE
                },
            ),
            Err(FixtureError::JsonFixtureSerialization { .. })
        ));
    }

    #[test]
    fn wire_fixture_yaml_serialization_enforces_the_loader_document_ceiling() {
        let source_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(source_file.path(), COMPLETE_FIXTURE).unwrap();
        let fixture = WireFixture::load(source_file.path()).unwrap();
        let diagnostic_path = Path::new("bounded-fixture.yaml");
        let document = serialize_yaml_with_limit(&fixture, diagnostic_path, usize::MAX).unwrap();

        assert_eq!(
            serialize_yaml_with_limit(&fixture, diagnostic_path, document.len()).unwrap(),
            document
        );
        assert!(matches!(
            serialize_yaml_with_limit(&fixture, diagnostic_path, document.len() - 1),
            Err(FixtureError::PersistedDocumentTooLarge { kind: "wire fixture" })
        ));
        assert!(matches!(
            serialize_yaml_with_limits(
                &fixture,
                diagnostic_path,
                usize::MAX,
                DocumentValidationLimits {
                    max_nodes: 1,
                    ..DocumentValidationLimits::TEST_PERMISSIVE
                },
            ),
            Err(FixtureError::YamlFixtureSerialization { .. })
        ));
    }

    #[test]
    fn yaml_preflight_rejects_a_typed_scalar_before_libyaml_serialization() {
        let source_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(source_file.path(), COMPLETE_FIXTURE).unwrap();
        let mut fixture = WireFixture::load(source_file.path()).unwrap();
        fixture.scenario_id = "x".repeat(2_048);

        let error = preflight_wire_fixture_for_yaml(
            &fixture,
            Path::new("bounded-fixture.yaml"),
            1_024,
            DocumentValidationLimits::TEST_PERMISSIVE,
        )
        .expect_err("typed scalars beyond the preflight ceiling must not reach libyaml");

        assert!(matches!(
            error,
            FixtureError::PersistedDocumentTooLarge { kind: "wire fixture" }
        ));
    }

    #[test]
    fn bounded_document_writer_never_reserves_past_its_document_limit() {
        let mut writer = BoundedDocumentWriter::new(10);

        writer.write_all(b"12345").unwrap();
        writer.write_all(b"6789").unwrap();

        assert!(writer.document.capacity() <= 10);
    }

    #[test]
    fn wire_fixture_write_rejects_an_unsupported_version() {
        // Arrange
        let source_file = tempfile::NamedTempFile::new().unwrap();
        let output_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(source_file.path(), COMPLETE_FIXTURE).unwrap();
        let mut fixture = WireFixture::load(source_file.path()).unwrap();
        fixture.version = 2;

        // Act
        let result = fixture.write(output_file.path());

        // Assert
        assert!(matches!(
            result,
            Err(FixtureError::UnsupportedWireFixtureVersion { version })
                if version == 2
        ));
    }

    #[test]
    fn wire_fixture_load_rejects_unknown_nested_fields() {
        // Arrange
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        let fixture = COMPLETE_FIXTURE.replace(
            "  source_id: capture-123\n",
            "  source_id: capture-123\n  unexpected: value\n",
        );
        fs::write(temporary_file.path(), fixture).unwrap();

        // Act
        let result = WireFixture::load(temporary_file.path());

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn wire_fixture_loads_and_writes_json_documents() {
        // Arrange
        let temporary_directory = tempfile::tempdir().unwrap();
        let source_path = temporary_directory.path().join("fixture.yaml");
        let output_path = temporary_directory.path().join("fixture.json");
        fs::write(&source_path, COMPLETE_FIXTURE).unwrap();
        let expected = WireFixture::load(&source_path).unwrap();

        // Act
        expected.write(&output_path).unwrap();
        let loaded = WireFixture::load(&output_path).unwrap();

        // Assert
        assert_eq!(loaded, expected);
        assert!(serde_json::from_slice::<Value>(&fs::read(output_path).unwrap()).is_ok());
    }

    #[test]
    fn inference_scenario_load_rejects_an_unsupported_version() {
        // Arrange
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(
            temporary_file.path(),
            COMPLETE_SCENARIO.replacen("version: 1", "version: 2", 1),
        )
        .unwrap();

        // Act
        let result = InferenceScenario::load(temporary_file.path());

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn scenarios_and_wire_fixtures_enforce_the_shared_turn_bound() {
        let scenario: InferenceScenario = serde_yaml::from_str(COMPLETE_SCENARIO).unwrap();
        let scenario_turn = scenario.turns[0].clone();
        let fixture: WireFixture = serde_yaml::from_str(COMPLETE_FIXTURE).unwrap();
        let fixture_turn = fixture.turns[0].clone();

        for count in [0, MAX_INFERENCE_TURNS + 1] {
            let mut candidate = scenario.clone();
            candidate.turns.resize(count, scenario_turn.clone());
            assert!(candidate.validate_version().is_err(), "scenario accepted {count} turns");

            let mut candidate = fixture.clone();
            candidate.turns.resize(count, fixture_turn.clone());
            assert!(candidate.validate_version().is_err(), "fixture accepted {count} turns");
        }
        for count in [1, MAX_INFERENCE_TURNS] {
            let mut candidate = scenario.clone();
            candidate.turns.resize(count, scenario_turn.clone());
            assert!(candidate.validate_version().is_ok(), "scenario rejected {count} turns");

            let mut candidate = fixture.clone();
            candidate.turns.resize(count, fixture_turn.clone());
            assert!(candidate.validate_version().is_ok(), "fixture rejected {count} turns");
        }
    }

    #[test]
    fn inference_scenario_repeatable_sse_events_are_backward_compatible_and_symmetric() {
        // Catches making the additive client/upstream repeat contract mandatory or one-sided.
        let legacy: InferenceScenario = serde_yaml::from_str(COMPLETE_SCENARIO).unwrap();
        let extended = COMPLETE_SCENARIO.replace(
            "      upstream_body_kind: json\n",
            "      upstream_body_kind: json\n      client_sse_events: [message_start, content_block_delta, message_stop]\n      client_sse_repeatable_events: [content_block_delta]\n      client_sse_interleaved_events: [ping]\n      upstream_sse_events: [response.delta]\n      upstream_sse_repeatable_events: [response.delta]\n      upstream_sse_interleaved_events: [ping]\n",
        );

        let extended: InferenceScenario = serde_yaml::from_str(&extended).unwrap();

        assert!(legacy.turns[0].expect.client_sse_repeatable_events.is_empty());
        assert!(legacy.turns[0].expect.client_sse_interleaved_events.is_empty());
        assert!(legacy.turns[0].expect.upstream_sse_repeatable_events.is_empty());
        assert!(legacy.turns[0].expect.upstream_sse_interleaved_events.is_empty());
        assert_eq!(
            extended.turns[0].expect.client_sse_repeatable_events,
            ["content_block_delta"]
        );
        assert_eq!(
            extended.turns[0].expect.upstream_sse_repeatable_events,
            ["response.delta"]
        );
        assert_eq!(extended.turns[0].expect.client_sse_interleaved_events, ["ping"]);
        assert_eq!(extended.turns[0].expect.upstream_sse_interleaved_events, ["ping"]);
    }

    #[test]
    fn inference_scenario_accepts_nonempty_client_and_upstream_sse_event_names() {
        // Catches rejecting representative named and repeatable client/upstream SSE declarations.
        let mut scenario: InferenceScenario = serde_yaml::from_str(COMPLETE_SCENARIO).unwrap();
        let expectation = &mut scenario.turns[0].expect;
        expectation.client_sse_events = event_names(&["message_start", "content_block_delta", "message_stop"]);
        expectation.client_sse_repeatable_events = event_names(&["content_block_delta"]);
        expectation.client_sse_interleaved_events = event_names(&["ping"]);
        expectation.upstream_sse_events = event_names(&["response.created", "response.delta", "response.completed"]);
        expectation.upstream_sse_repeatable_events = event_names(&["response.delta"]);
        expectation.upstream_sse_interleaved_events = event_names(&["ping"]);

        let result = scenario.validate_version();

        assert!(result.is_ok());
    }

    #[test]
    fn sse_declaration_index_counts_occurrences_and_borrows_repeatable_names() {
        let expected = event_names(&["message_start", "content_block_delta", "content_block_delta"]);
        let repeatable = event_names(&["content_block_delta"]);

        let index = SseDeclarationIndex::new(&expected, &repeatable);

        assert_eq!(index.ordered_occurrences.get("message_start"), Some(&1));
        assert_eq!(index.ordered_occurrences.get("content_block_delta"), Some(&2));
        let indexed_repeatable = index.repeatable.get("content_block_delta").unwrap();
        assert_eq!(indexed_repeatable.as_ptr(), repeatable[0].as_ptr());
    }

    #[test]
    fn inference_scenario_accepts_minimum_nonempty_sse_event_names_and_empty_lists() {
        // Catches treating an empty list or the shortest nonempty event name as an empty entry.
        let empty: InferenceScenario = serde_yaml::from_str(COMPLETE_SCENARIO).unwrap();
        let mut minimum: InferenceScenario = serde_yaml::from_str(COMPLETE_SCENARIO).unwrap();
        let expectation = &mut minimum.turns[0].expect;
        expectation.client_sse_events = event_names(&["x"]);
        expectation.client_sse_repeatable_events = event_names(&["x"]);
        expectation.client_sse_interleaved_events = event_names(&["ping"]);
        expectation.upstream_sse_events = event_names(&["y"]);
        expectation.upstream_sse_repeatable_events = event_names(&["y"]);
        expectation.upstream_sse_interleaved_events = event_names(&["pong"]);

        assert!(empty.validate_version().is_ok());
        assert!(minimum.validate_version().is_ok());
    }

    #[test]
    fn inference_scenario_rejects_empty_entries_in_every_sse_event_name_list() {
        // Catches omitting nonempty validation from any named/repeatable client/upstream list.
        for field in [
            "client_sse_events",
            "client_sse_repeatable_events",
            "client_sse_interleaved_events",
            "upstream_sse_events",
            "upstream_sse_repeatable_events",
            "upstream_sse_interleaved_events",
        ] {
            let mut scenario: InferenceScenario = serde_yaml::from_str(COMPLETE_SCENARIO).unwrap();
            sse_event_names_mut(&mut scenario.turns[0].expect, field).push(String::new());

            let result = scenario.validate_version();
            let expected_path = format!("turns[0].expect.{field}");

            assert!(matches!(
                result,
                Err(FixtureError::InvalidScenarioExpectation { path, rule })
                    if path == expected_path && rule == "SSE event name must not be empty"
            ));
        }
    }

    fn event_names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn sse_event_names_mut<'a>(expectation: &'a mut ScenarioExpectation, field: &str) -> &'a mut Vec<String> {
        match field {
            "client_sse_events" => &mut expectation.client_sse_events,
            "client_sse_repeatable_events" => &mut expectation.client_sse_repeatable_events,
            "client_sse_interleaved_events" => &mut expectation.client_sse_interleaved_events,
            "upstream_sse_events" => &mut expectation.upstream_sse_events,
            "upstream_sse_repeatable_events" => &mut expectation.upstream_sse_repeatable_events,
            "upstream_sse_interleaved_events" => &mut expectation.upstream_sse_interleaved_events,
            _ => panic!("unknown SSE event-name field"),
        }
    }

    #[test]
    fn inference_scenario_rejects_invalid_interleaved_sse_metadata() {
        // Catches ambiguous or silently deduplicated optional interleaved declarations.
        let invalid = [
            (
                "duplicate declaration",
                "      client_sse_interleaved_events: [ping, ping]\n",
            ),
            (
                "overlap with ordered declaration",
                "      client_sse_events: [message_start, ping, message_stop]\n      client_sse_interleaved_events: [ping]\n",
            ),
            (
                "overlap with repeatable declaration",
                "      client_sse_events: [message_start, content_block_delta, message_stop]\n      client_sse_repeatable_events: [content_block_delta]\n      client_sse_interleaved_events: [content_block_delta]\n",
            ),
            (
                "upstream overlap",
                "      upstream_sse_events: [message_start, message_stop]\n      upstream_sse_interleaved_events: [message_start]\n",
            ),
        ];

        for (case, metadata) in invalid {
            let temporary_file = tempfile::NamedTempFile::new().unwrap();
            let scenario = COMPLETE_SCENARIO.replace(
                "      upstream_body_kind: json\n",
                &format!("      upstream_body_kind: json\n{metadata}"),
            );
            fs::write(temporary_file.path(), scenario).unwrap();

            let error = InferenceScenario::load(temporary_file.path())
                .expect_err("invalid interleaved SSE metadata must fail scenario loading");

            assert!(error.to_string().contains("interleaved SSE event"), "{case}: {error}");
        }
    }

    #[test]
    fn inference_scenario_rejects_invalid_repeatable_sse_metadata() {
        // Catches silently ignoring duplicate, unknown, or ambiguous repeat declarations.
        let invalid = [
            (
                "duplicate declaration",
                "      client_sse_events: [message_start, content_block_delta, message_stop]\n      client_sse_repeatable_events: [content_block_delta, content_block_delta]\n",
            ),
            (
                "unknown declaration",
                "      client_sse_events: [message_start, message_stop]\n      client_sse_repeatable_events: [content_block_delta]\n",
            ),
            (
                "ambiguous declaration",
                "      client_sse_events: [content_block_delta, content_block_delta]\n      client_sse_repeatable_events: [content_block_delta]\n",
            ),
        ];

        for (case, metadata) in invalid {
            let temporary_file = tempfile::NamedTempFile::new().unwrap();
            let scenario = COMPLETE_SCENARIO.replace(
                "      upstream_body_kind: json\n",
                &format!("      upstream_body_kind: json\n{metadata}"),
            );
            fs::write(temporary_file.path(), scenario).unwrap();

            let error = InferenceScenario::load(temporary_file.path())
                .expect_err("invalid repeatable SSE metadata must fail scenario loading");

            assert!(error.to_string().contains("repeatable SSE event"), "{case}: {error}");
        }
    }

    #[test]
    fn recorded_body_round_trips_json_content_types() {
        // Arrange
        let input = br#"{"id":"response-1","output":["hello"]}"#;

        // Act
        let body = RecordedBody::from_http(Some("application/problem+json; charset=utf-8"), input).unwrap();
        let rendered = body.render().unwrap();

        // Assert
        assert_eq!(body.kind(), BodyKind::Json);
        assert_eq!(
            serde_json::from_slice::<Value>(&rendered).unwrap(),
            json!({"id": "response-1", "output": ["hello"]})
        );
    }

    #[test]
    fn recorded_body_parses_non_empty_application_json() {
        // Arrange
        let input = br#"{"request":"value"}"#;

        // Act
        let body = RecordedBody::from_http(Some("application/json"), input).unwrap();

        // Assert
        assert_eq!(
            body,
            RecordedBody::Json {
                value: json!({"request": "value"}),
            }
        );
    }

    #[test]
    fn recorded_body_round_trips_binary_content_as_base64() {
        // Arrange
        let input = [0_u8, 255, 17, 42];

        // Act
        let body = RecordedBody::from_http(Some("application/octet-stream"), &input).unwrap();
        let rendered = body.render().unwrap();

        // Assert
        assert_eq!(body.kind(), BodyKind::Base64);
        assert_eq!(rendered, input);
    }

    #[test]
    fn recorded_body_uses_standard_base64_wire_text() {
        // Arrange
        let input = [0_u8, 255, 17, 42];

        // Act
        let body = RecordedBody::from_http(Some("application/octet-stream"), &input).unwrap();

        // Assert
        assert_eq!(
            body,
            RecordedBody::Base64 {
                data: "AP8RKg==".to_owned(),
            }
        );
    }

    #[test]
    fn recorded_body_parses_empty_http_bodies_before_consulting_content_type() {
        // Arrange
        let input = [];

        // Act
        let body = RecordedBody::from_http(Some("application/json"), &input).unwrap();

        // Assert
        assert_eq!(body, RecordedBody::Empty);
        assert_eq!(body.kind(), BodyKind::Empty);
        assert_eq!(body.render().unwrap(), input);
    }

    #[test]
    fn recorded_body_parses_multiline_sse_metadata_and_done_marker() {
        // Arrange
        let input = b"event: response.output_text.delta\r\nid: event-7\r\nretry: 2500\r\ndata: first line\r\ndata: second line\r\n\r\ndata: [DONE]\r\n\r\n";

        // Act
        let body = RecordedBody::from_http(Some("text/event-stream; charset=utf-8"), input).unwrap();

        // Assert
        assert_eq!(
            body,
            RecordedBody::Sse {
                frames: vec![SseFrame {
                    event: Some("response.output_text.delta".to_owned()),
                    data: "first line\nsecond line".to_owned(),
                    id: Some("event-7".to_owned()),
                    retry: Some(2500),
                }],
                done: true,
            }
        );
    }

    #[test]
    fn recorded_body_renders_metadata_rich_sse_canonically() {
        // Arrange
        let input = b"event: response.output_text.delta\r\nid: event-7\r\nretry: 2500\r\ndata: first line\r\ndata: second line\r\n\r\ndata: [DONE]\r\n\r\n";

        // Act
        let body = RecordedBody::from_http(Some("text/event-stream"), input).unwrap();
        let rendered = body.render().unwrap();

        // Assert
        assert_eq!(
        rendered,
        b"event: response.output_text.delta\ndata: first line\ndata: second line\nid: event-7\nretry: 2500\n\ndata: [DONE]\n\n"
    );
    }

    #[test]
    fn recorded_body_renders_canonical_sse_and_flushes_an_unterminated_frame() {
        // Arrange
        let input = b"event: update\ndata: complete";

        // Act
        let parsed = RecordedBody::from_http(Some("text/event-stream"), input).unwrap();
        let rendered = parsed.render().unwrap();

        // Assert
        assert_eq!(
            parsed,
            RecordedBody::Sse {
                frames: vec![SseFrame {
                    event: Some("update".to_owned()),
                    data: "complete".to_owned(),
                    id: None,
                    retry: None,
                }],
                done: false,
            }
        );
        assert_eq!(rendered, b"event: update\ndata: complete\n\n");
    }

    #[test]
    fn recorded_body_renders_done_marker_only_when_recorded() {
        // Arrange
        let body = RecordedBody::Sse {
            frames: Vec::new(),
            done: true,
        };

        // Act
        let rendered = body.render().unwrap();

        // Assert
        assert_eq!(rendered, b"data: [DONE]\n\n");
    }

    #[test]
    fn recorded_body_ignores_data_after_the_terminal_done_marker() {
        // Arrange
        let input = b"event: response.started\ndata: before\n\ndata: [DONE]\n\nevent: response.delta\ndata: after\n\n";

        // Act
        let body = RecordedBody::from_http(Some("text/event-stream"), input).unwrap();
        let rendered = body.render().unwrap();

        // Assert
        assert_eq!(
            body,
            RecordedBody::Sse {
                frames: vec![SseFrame {
                    event: Some("response.started".to_owned()),
                    data: "before".to_owned(),
                    id: None,
                    retry: None,
                }],
                done: true,
            }
        );
        assert_eq!(rendered, b"event: response.started\ndata: before\n\ndata: [DONE]\n\n");
    }

    #[test]
    fn recorded_body_treats_done_line_in_multiline_event_as_ordinary_data() {
        let input = b"event: update\ndata: [DONE]\ndata: still-data\n\ndata: [DONE]\n\n";

        let body = RecordedBody::from_http(Some("text/event-stream"), input).unwrap();

        assert_eq!(
            body,
            RecordedBody::Sse {
                frames: vec![SseFrame {
                    event: Some("update".to_owned()),
                    data: "[DONE]\nstill-data".to_owned(),
                    id: None,
                    retry: None,
                }],
                done: true,
            }
        );
    }

    #[test]
    fn bounded_sse_parser_accepts_exact_limits_and_rejects_first_excess() {
        let limits = SseParseLimits {
            max_frames: 2,
            max_frame_bytes: b"data: x\n\n".len(),
        };
        let exact = b"data: x\n\ndata: x\n\n";
        let count_plus_one = b"data: x\n\ndata: x\n\ndata: x\n\n";
        let frame_plus_one = b"data: xx\n\n";

        RecordedBody::from_http_with_sse_limits(Some("text/event-stream"), exact, limits).unwrap();
        assert!(RecordedBody::from_http_with_sse_limits(Some("text/event-stream"), count_plus_one, limits).is_err());
        assert!(RecordedBody::from_http_with_sse_limits(Some("text/event-stream"), frame_plus_one, limits).is_err());
        assert!(RecordedBody::from_http_with_sse_limits(Some("text/event-stream"), b"data: \xff\n\n", limits).is_err());
    }

    #[test]
    fn rejected_sse_data_leaves_borrowed_line_storage_and_counters_unchanged() {
        let tiny = SseParseLimits {
            max_frames: 1,
            max_frame_bytes: b"data: \n\n".len(),
        };
        let mut builder = SseFrameBuilder::default();
        let initial_capacity = builder.data_lines.capacity();

        assert!(record_sse_field(&mut builder, "data", "x", 0, Some(tiny)).is_err());
        assert_eq!(builder.data_lines.len(), 0);
        assert_eq!(builder.data_lines.capacity(), initial_capacity);
        assert_eq!(builder.field_bytes, 0);
    }

    #[test]
    fn rejected_sse_singleton_fields_leave_accepted_frame_state_unchanged() {
        let tiny = SseParseLimits {
            max_frames: 1,
            max_frame_bytes: b"data: \n\n".len(),
        };
        let generous = SseParseLimits {
            max_frames: 1,
            max_frame_bytes: usize::MAX,
        };
        let mut builder = SseFrameBuilder::default();
        record_sse_field(&mut builder, "data", "", 0, Some(generous)).unwrap();
        let accepted_bytes = builder.field_bytes;
        let accepted_capacity = builder.data_lines.capacity();

        assert!(record_sse_field(&mut builder, "event", "x", 0, Some(tiny)).is_err());
        assert_eq!((builder.event, builder.event_bytes), (None, 0));
        assert_eq!(builder.field_bytes, accepted_bytes);
        assert_eq!(builder.data_lines.capacity(), accepted_capacity);

        assert!(record_sse_field(&mut builder, "id", "x", 0, Some(tiny)).is_err());
        assert_eq!((builder.id, builder.id_bytes), (None, 0));
        assert_eq!(builder.field_bytes, accepted_bytes);
        assert_eq!(builder.data_lines.capacity(), accepted_capacity);
    }

    #[test]
    fn overflowing_sse_data_size_leaves_borrowed_line_storage_unchanged() {
        let generous = SseParseLimits {
            max_frames: 1,
            max_frame_bytes: usize::MAX,
        };
        let mut builder = SseFrameBuilder {
            field_bytes: usize::MAX,
            ..SseFrameBuilder::default()
        };
        let initial_capacity = builder.data_lines.capacity();

        assert!(record_sse_field(&mut builder, "data", "", 0, Some(generous)).is_err());
        assert_eq!(builder.field_bytes, usize::MAX);
        assert_eq!(builder.data_lines.len(), 0);
        assert_eq!(builder.data_lines.capacity(), initial_capacity);
    }

    #[test]
    fn inference_scenario_bind_model_replaces_only_exact_json_string_values() {
        // Arrange
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(temporary_file.path(), COMPLETE_SCENARIO).unwrap();
        let scenario = InferenceScenario::load(temporary_file.path()).unwrap();

        // Act
        let bound = scenario.bind_model("model-under-test");

        // Assert
        let RecordedBody::Json { value } = &bound.turns[0].request.body else {
            panic!("the scenario request should have a JSON body");
        };
        assert_eq!(value["model"], "model-under-test");
        assert_eq!(value["nested"]["models"][0], "model-under-test");
        assert_eq!(value["nested"]["models"][1], "prefix-${MODEL}");
        let RecordedBody::Json { value: original } = &scenario.turns[0].request.body else {
            panic!("the original scenario request should have a JSON body");
        };
        assert_eq!(original["model"], "${MODEL}");
    }

    #[test]
    fn scenario_turn_binds_only_exact_previous_response_id_values() {
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(temporary_file.path(), COMPLETE_SCENARIO).unwrap();
        let mut scenario = InferenceScenario::load(temporary_file.path()).unwrap();
        let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
            panic!("the scenario request should have a JSON body");
        };
        value["previous_response_id"] = json!("${PREVIOUS_RESPONSE_ID}");
        value["nested"]["literal"] = json!("prefix-${PREVIOUS_RESPONSE_ID}");

        scenario.turns[0]
            .bind_previous_response_id(Some("resp_previous"))
            .unwrap();

        let RecordedBody::Json { value } = &scenario.turns[0].request.body else {
            panic!("the scenario request should have a JSON body");
        };
        assert_eq!(value["previous_response_id"], "resp_previous");
        assert_eq!(value["nested"]["literal"], "prefix-${PREVIOUS_RESPONSE_ID}");
    }

    #[test]
    fn scenario_turn_rejects_previous_response_placeholder_without_prior_response() {
        let temporary_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(temporary_file.path(), COMPLETE_SCENARIO).unwrap();
        let mut scenario = InferenceScenario::load(temporary_file.path()).unwrap();
        let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
            panic!("the scenario request should have a JSON body");
        };
        value["previous_response_id"] = json!("${PREVIOUS_RESPONSE_ID}");

        let error = scenario.turns[0]
            .bind_previous_response_id(None)
            .expect_err("the first turn cannot reference a preceding response");

        assert_eq!(
            error.to_string(),
            "scenario previous response ID placeholder has no preceding response"
        );
    }
}
