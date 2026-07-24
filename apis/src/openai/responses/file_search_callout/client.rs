// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Callout client for OGX vector store search.

use std::{
    fmt, io,
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use http::{HeaderMap, HeaderValue};
use praxis_core::callout::FailureMode;
use serde::{
    Deserialize, Serialize,
    de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;

use crate::openai::api_client::ApiClient;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default number of retained results per file-search call.
const DEFAULT_MAX_NUM_RESULTS: usize = 10;

/// Maximum number of concurrent callouts to OGX.
pub(super) const MAX_CONCURRENT_SEARCHES: usize = 8;

/// OpenAI's maximum number of file-search results per call.
const MAX_NUM_RESULTS: usize = 50;

/// Maximum content chunks decoded across one response page.
const MAX_DECODED_CONTENT_CHUNKS: usize = 4_096;

/// Maximum original queries accepted in response page metadata.
const MAX_QUERIES_PER_RESPONSE: usize = 64;

/// Maximum opaque pagination token bytes accepted from OGX.
const MAX_NEXT_PAGE_BYTES: usize = 4_096;

/// Maximum bytes accepted in one decoded content chunk.
const MAX_CONTENT_CHUNK_BYTES: usize = 2_097_152; // 2 MiB

/// Maximum entries in OpenAI-compatible result attributes.
const MAX_ATTRIBUTE_ENTRIES: usize = 16;

/// Maximum characters in one OpenAI-compatible attribute key.
const MAX_ATTRIBUTE_KEY_CHARS: usize = 64;

/// Maximum characters in one OpenAI-compatible string attribute value.
const MAX_ATTRIBUTE_STRING_CHARS: usize = 512;

/// Maximum nesting accepted in ignored OGX content extensions.
const MAX_IGNORED_EXTENSION_DEPTH: usize = 8;

/// Maximum entries accepted in one ignored OGX extension container.
const MAX_IGNORED_EXTENSION_ENTRIES: usize = 65_536;

/// Maximum bytes accepted in one ignored OGX extension string or key.
const MAX_IGNORED_EXTENSION_STRING_BYTES: usize = 2_097_152; // 2 MiB

/// Maximum bytes accepted in a schema field name before it is owned.
const MAX_SCHEMA_FIELD_BYTES: usize = 64;

/// Maximum rendered query size sent to OGX: 64 `KiB`.
pub(super) const MAX_QUERY_BYTES: usize = 65_536;

/// Maximum serialized request body sent to OGX: 1 MiB.
pub(super) const MAX_SEARCH_REQUEST_BYTES: usize = 1_048_576;

/// Maximum vector store identifier size accepted for one URL segment.
pub(super) const MAX_VECTOR_STORE_ID_BYTES: usize = 512;

/// Allocation unit used by global collected-response admission.
const RESPONSE_BODY_BUDGET_UNIT_BYTES: usize = 1_048_576; // 1 MiB

/// Charge both the collected body and its decoded representation.
const RESPONSE_DECODE_MEMORY_MULTIPLIER: usize = 2;

/// Process-wide response bytes reserved across all configured clients.
const GLOBAL_RESPONSE_BODY_BUDGET_UNITS: usize = 512; // 512 MiB

/// Process-wide blocking decoder slots, including timed-out tasks.
const GLOBAL_RESPONSE_DECODE_SLOTS: usize = 32;

/// Fair byte-weighted admission before any response body is collected.
static RESPONSE_BODY_BUDGET: LazyLock<Arc<tokio::sync::Semaphore>> =
    LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(GLOBAL_RESPONSE_BODY_BUDGET_UNITS)));

/// Fair admission for blocking decoders across filter instances.
static RESPONSE_DECODE_SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(GLOBAL_RESPONSE_DECODE_SLOTS);

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// Specification for a single search request.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SearchSpec<'a> {
    /// Index of the file-search call that owns this request.
    pub call_index: usize,

    /// Filter criteria.
    pub filters: Option<&'a Value>,

    /// Maximum number of aggregated results.
    pub max_num_results: Option<u64>,

    /// Search query.
    pub query: &'a str,

    /// Ranking options.
    pub ranking_options: Option<&'a Value>,

    /// Vector store ID.
    pub store_id: &'a str,
}

/// Request to search a vector store.
#[derive(Debug, Serialize)]
pub(crate) struct VectorStoreSearchRequest<'a> {
    /// Filter criteria.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) filters: Option<&'a Value>,

    /// Maximum number of results to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_num_results: Option<u64>,

    /// Search query.
    pub(crate) query: &'a str,

    /// OGX ranking options translated from the Responses shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ranking_options: Option<OgxSearchRankingOptions<'a>>,

    /// OGX must not perform a second query rewrite.
    pub(crate) rewrite_query: bool,

    /// Enable OGX keyword and vector retrieval for hybrid ranking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_mode: Option<&'static str>,
}

impl<'a> VectorStoreSearchRequest<'a> {
    /// Build the exact OGX wire model from Responses ranking options.
    pub(super) fn new(
        filters: Option<&'a Value>,
        max_num_results: Option<u64>,
        query: &'a str,
        responses_ranking_options: Option<&'a Value>,
    ) -> Result<Self, &'static str> {
        let translated = translate_ranking_options(responses_ranking_options)?;
        Ok(Self {
            filters,
            max_num_results,
            query,
            ranking_options: translated.options,
            rewrite_query: false,
            search_mode: translated.search_mode,
        })
    }
}

/// Ranking options accepted by OGX's vector-store search endpoint.
#[derive(Debug, Serialize)]
pub(crate) struct OgxSearchRankingOptions<'a> {
    /// Explicit ranker, or OGX's weighted ranker for hybrid weights.
    #[serde(skip_serializing_if = "Option::is_none")]
    ranker: Option<&'a str>,

    /// Normalized semantic weight understood by OGX's weighted ranker.
    #[serde(skip_serializing_if = "Option::is_none")]
    alpha: Option<f64>,

    /// Minimum score copied from the Responses request.
    #[serde(skip_serializing_if = "Option::is_none")]
    score_threshold: Option<&'a Value>,
}

/// Borrowed ranking translation plus the OGX retrieval mode it requires.
struct TranslatedRankingOptions<'a> {
    /// Translated ranking options.
    options: Option<OgxSearchRankingOptions<'a>>,

    /// OGX retrieval mode.
    search_mode: Option<&'static str>,
}

/// Response from vector store search.
#[cfg(test)]
#[derive(Debug, Deserialize)]
pub(crate) struct VectorStoreSearchResponse {
    /// Search results.
    pub data: Vec<SearchResult>,
}

/// Single search result from a vector store.
#[derive(Debug, Deserialize)]
pub(crate) struct SearchResult {
    /// Optional attributes.
    #[serde(default)]
    pub attributes: Option<Value>,

    /// Content chunks.
    pub content: Vec<ContentChunk>,

    /// File ID.
    pub file_id: String,

    /// Filename.
    pub filename: String,

    /// Relevance score.
    pub score: f64,
}

/// Content chunk within a search result.
#[derive(Debug, Deserialize)]
pub(crate) struct ContentChunk {
    /// Chunk type.
    #[serde(rename = "type")]
    pub _chunk_type: ContentChunkType,

    /// Chunk text.
    pub text: String,
}

/// Supported vector-store content chunk type.
#[derive(Debug)]
pub(crate) enum ContentChunkType {
    /// Plain text chunk.
    Text,
}

impl<'de> Deserialize<'de> for ContentChunkType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(ContentChunkTypeVisitor)
    }
}

/// Visitor that recognizes a content type without echoing rejected input.
struct ContentChunkTypeVisitor;

impl Visitor<'_> for ContentChunkTypeVisitor {
    type Value = ContentChunkType;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the supported vector-store content chunk type")
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        if v == "text" {
            Ok(ContentChunkType::Text)
        } else {
            Err(E::custom("unsupported vector-store content chunk type"))
        }
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        self.visit_str(&v)
    }
}

/// Errors from file search callouts.
#[derive(Debug)]
pub(crate) enum FileSearchError {
    /// Callout failed (network, timeout, policy, or HTTP error).
    Callout {
        /// Error message.
        message: String,

        /// Vector store ID.
        store_id: String,
    },

    /// Response deserialization failed.
    Deserialize {
        /// Bytes collected from the successful HTTP response.
        body_bytes: usize,

        /// One-indexed line where decoding failed.
        line: usize,

        /// One-indexed column where decoding failed.
        column: usize,

        /// Vector store ID.
        store_id: String,
    },
}

impl fmt::Display for FileSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Callout { message, store_id } => {
                write!(f, "callout to store {store_id:?} failed: {message}")
            },
            Self::Deserialize {
                body_bytes: _,
                line,
                column,
                store_id,
            } => {
                write!(
                    f,
                    "invalid vector-store response from store {store_id:?} at line {line}, column {column}"
                )
            },
        }
    }
}

impl std::error::Error for FileSearchError {}

/// One failed vector-store search.
#[derive(Debug)]
pub(crate) struct SearchFailure {
    /// Index of the file-search call that owns the failed request.
    pub call_index: usize,

    /// Failure details.
    pub error: FileSearchError,
}

/// Incrementally aggregated result of a search fan-out.
#[derive(Debug)]
pub(crate) struct SearchBatch {
    /// Individual request failures.
    pub failures: Vec<SearchFailure>,

    /// Ranked, bounded results for each file-search call.
    pub results_by_call: Vec<Vec<SearchResult>>,

    /// Process-wide response admission retained while decoded payloads live.
    response_admission: Option<Arc<ResponseAdmission>>,
}

impl SearchBatch {
    /// Create an empty batch for the planned calls.
    pub(super) fn new(call_count: usize) -> Self {
        Self {
            failures: Vec::new(),
            results_by_call: (0..call_count).map(|_| Vec::new()).collect(),
            response_admission: None,
        }
    }

    /// Create an empty result batch carrying planning failures.
    pub(super) fn with_failures(call_count: usize, failures: Vec<SearchFailure>) -> Self {
        Self {
            failures,
            ..Self::new(call_count)
        }
    }

    /// Sort every retained result set by descending score.
    fn sort_results(&mut self) {
        for results in &mut self.results_by_call {
            results.sort_by(|left, right| right.score.total_cmp(&left.score));
        }
    }
}

/// Construction parameters for [`FileSearchClient`].
pub(crate) struct FileSearchClientConfig {
    /// Shared OpenAI-compatible API client.
    pub api_client: ApiClient,

    /// Optional prevalidated authorization value.
    pub authorization: Option<HeaderValue>,

    /// Whether one failed chunk stops scheduling later callouts.
    pub failure_mode: FailureMode,

    /// Maximum response body size enforced by the core client.
    pub max_response_bytes: usize,

    /// Maximum successful response bytes retained across the fan-out.
    pub max_total_response_bytes: usize,

    /// Whole-call timeout, including response body collection.
    pub timeout: Duration,
}

/// Client for OGX vector store search API.
pub(crate) struct FileSearchClient {
    /// Shared OpenAI-compatible API client.
    api_client: ApiClient,

    /// Optional prevalidated authorization value.
    authorization: Option<HeaderValue>,

    /// Whether one failed chunk stops scheduling later callouts.
    failure_mode: FailureMode,

    /// Maximum response body size enforced by the core client.
    max_response_bytes: usize,

    /// Maximum successful response bytes retained across the fan-out.
    max_total_response_bytes: usize,

    /// Whole-call timeout, including response body collection.
    timeout: Duration,
}

impl FileSearchClient {
    /// Create a new client.
    pub fn new(config: FileSearchClientConfig) -> Self {
        Self {
            api_client: config.api_client,
            authorization: config.authorization,
            failure_mode: config.failure_mode,
            max_response_bytes: config.max_response_bytes,
            max_total_response_bytes: config.max_total_response_bytes,
            timeout: config.timeout,
        }
    }

    /// Search multiple vector stores with bounded concurrency and aggregation.
    #[expect(
        clippy::too_many_lines,
        reason = "deadline, budget, and failure policy share one scheduling loop"
    )]
    pub async fn search(&self, specs: &[SearchSpec<'_>], call_count: usize) -> SearchBatch {
        let mut batch = SearchBatch::new(call_count);
        let mut consumed_response_bytes = 0_usize;
        let mut deadline_recorded = false;
        let mut next_spec = 0_usize;
        let execution_started = Instant::now();
        let admission = match self.acquire_execution_admission(specs.len(), execution_started).await {
            Ok(admission) => admission,
            Err(message) => {
                append_admission_failures(&mut batch.failures, specs, message);
                return batch;
            },
        };
        batch.response_admission = Some(Arc::clone(&admission));

        while next_spec < specs.len() {
            if execution_started.elapsed() >= self.timeout {
                if let Some(remaining_specs) = specs.get(next_spec..) {
                    append_deadline_failures(&mut batch.failures, remaining_specs);
                }
                break;
            }

            let chunk_len = self.reserved_chunk_len(consumed_response_bytes, specs.len() - next_spec);
            if chunk_len == 0 {
                if let Some(remaining_specs) = specs.get(next_spec..) {
                    append_budget_failures(&mut batch.failures, remaining_specs, self.max_total_response_bytes);
                }
                break;
            }

            let Some(chunk) = specs.get(next_spec..next_spec.saturating_add(chunk_len)) else {
                break;
            };
            let futures = chunk
                .iter()
                .map(|spec| self.search_one(spec, execution_started, Arc::clone(&admission)));
            let chunk_results = futures::future::join_all(futures).await;
            let chunk_failed = merge_chunk_results(
                &mut batch,
                &mut consumed_response_bytes,
                chunk,
                chunk_results,
                self.max_total_response_bytes,
            );

            next_spec = next_spec.saturating_add(chunk_len);
            if execution_started.elapsed() >= self.timeout {
                let chunk_start = next_spec.saturating_sub(chunk_len);
                if let Some(current_and_remaining) = specs.get(chunk_start..) {
                    append_deadline_failures(&mut batch.failures, current_and_remaining);
                }
                deadline_recorded = true;
                break;
            }
            if chunk_failed && self.failure_mode == FailureMode::Closed {
                if let Some(remaining_specs) = specs.get(next_spec..) {
                    append_fail_closed_failures(&mut batch.failures, remaining_specs);
                }
                break;
            }
        }

        batch.sort_results();
        if !deadline_recorded && execution_started.elapsed() >= self.timeout {
            append_deadline_failures(&mut batch.failures, specs);
        }
        batch
    }

    /// Search a single vector store.
    async fn search_one(
        &self,
        spec: &SearchSpec<'_>,
        execution_started: Instant,
        response_admission: Arc<ResponseAdmission>,
    ) -> Result<SearchResponse, FileSearchError> {
        deadline_remaining(self.timeout, execution_started, spec.store_id)?;
        let request = self.build_request(spec, execution_started)?;
        deadline_remaining(self.timeout, execution_started, spec.store_id)?;
        let body = self.execute_request(request, spec.store_id, execution_started).await?;
        parse_response_body_with_deadline(
            body,
            spec.store_id,
            result_limit(spec.max_num_results),
            execution_started,
            self.timeout,
            response_admission,
        )
        .await
    }

    /// Reserve the execution's aggregate response budget before fan-out.
    async fn acquire_execution_admission(
        &self,
        spec_count: usize,
        execution_started: Instant,
    ) -> Result<Arc<ResponseAdmission>, &'static str> {
        let aggregate_units =
            response_admission_units(self.max_response_bytes, self.max_total_response_bytes, spec_count)?;
        let remaining = self
            .timeout
            .checked_sub(execution_started.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or("file-search execution deadline exceeded while waiting for response admission")?;
        let body_budget = tokio::time::timeout(
            remaining,
            Arc::clone(&RESPONSE_BODY_BUDGET).acquire_many_owned(aggregate_units),
        )
        .await
        .map_err(|_elapsed| "file-search execution deadline exceeded while waiting for response admission")?
        .map_err(|_closed| "response body admission is unavailable")?;
        Ok(Arc::new(ResponseAdmission {
            _body_budget: body_budget,
        }))
    }

    /// Build one owned API request from borrowed search inputs.
    fn build_request(
        &self,
        spec: &SearchSpec<'_>,
        execution_started: Instant,
    ) -> Result<PreparedSearchRequest, FileSearchError> {
        let url = self.search_url(spec.store_id)?;
        if spec.query.len() > MAX_QUERY_BYTES {
            return Err(request_error(
                spec.store_id,
                format!("search query exceeds {MAX_QUERY_BYTES} byte limit"),
            ));
        }
        deadline_remaining(self.timeout, execution_started, spec.store_id)?;
        let request_body =
            VectorStoreSearchRequest::new(spec.filters, spec.max_num_results, spec.query, spec.ranking_options)
                .map_err(|message| request_error(spec.store_id, message))?;
        let body = serialize_bounded_request(&request_body, spec.store_id, execution_started, self.timeout)?;

        Ok(PreparedSearchRequest { body, url })
    }

    /// Execute a callout with a deadline covering body collection.
    async fn execute_request(
        &self,
        request: PreparedSearchRequest,
        store_id: &str,
        execution_started: Instant,
    ) -> Result<Vec<u8>, FileSearchError> {
        let remaining = deadline_remaining(self.timeout, execution_started, store_id)?;
        let headers = self.request_headers();
        tokio::time::timeout(
            remaining,
            self.api_client.post_json_bytes(request.url, request.body, &headers),
        )
        .await
        .map_err(|_elapsed| execution_deadline_error(store_id))
        .and_then(|result| result.map_err(|error| request_error(store_id, error.to_string())))
    }

    /// Build owned headers required by one callout request.
    fn request_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(authorization) = &self.authorization {
            headers.insert(http::header::AUTHORIZATION, authorization.clone());
        }
        headers
    }

    /// Calculate a chunk whose worst-case bodies fit the remaining budget.
    fn reserved_chunk_len(&self, consumed_bytes: usize, remaining_specs: usize) -> usize {
        let remaining_bytes = self.max_total_response_bytes.saturating_sub(consumed_bytes);
        (remaining_bytes / self.max_response_bytes)
            .min(MAX_CONCURRENT_SEARCHES)
            .min(remaining_specs)
    }

    /// Build a search URL without treating a store ID as path syntax.
    fn search_url(&self, store_id: &str) -> Result<String, FileSearchError> {
        if store_id.len() > MAX_VECTOR_STORE_ID_BYTES {
            return Err(request_error(
                store_id,
                format!("vector store ID exceeds {MAX_VECTOR_STORE_ID_BYTES} byte limit"),
            ));
        }
        if store_id.is_empty() || matches!(store_id, "." | "..") {
            return Err(request_error(
                store_id,
                "vector store ID must be a non-empty path segment",
            ));
        }

        self.api_client
            .resource_url("v1/vector_stores", store_id, Some("search"))
            .map_err(|error| request_error(store_id, error.to_string()))
    }
}

// -----------------------------------------------------------------------------
// Private types
// -----------------------------------------------------------------------------

/// A parsed response and its bounded wire size.
#[derive(Debug)]
struct SearchResponse {
    /// Successful response bytes consumed from the aggregate budget.
    body_bytes: usize,

    /// Parsed search results.
    data: Vec<SearchResult>,
}

/// Prepared bounded request passed to the shared API client.
struct PreparedSearchRequest {
    /// Pre-serialized JSON request body.
    body: Vec<u8>,

    /// Fully encoded OGX search URL.
    url: String,
}

/// Process-wide admission retained through aggregation and timed-out decoders.
#[derive(Debug)]
struct ResponseAdmission {
    /// Reserved aggregate-response byte units.
    _body_budget: tokio::sync::OwnedSemaphorePermit,
}

/// JSON writer that never retains more than the outbound request limit.
struct BoundedRequestWriter {
    /// Serialized bytes retained so far.
    body: Vec<u8>,

    /// Whether serialization attempted to cross the limit.
    exceeded: bool,

    /// Whether serialization crossed the shared execution deadline.
    deadline_elapsed: bool,

    /// Start of the shared execution deadline.
    execution_started: Instant,

    /// Total shared execution duration.
    timeout: Duration,
}

impl BoundedRequestWriter {
    /// Create an empty bounded request buffer.
    fn new(execution_started: Instant, timeout: Duration) -> Self {
        Self {
            body: Vec::new(),
            exceeded: false,
            deadline_elapsed: false,
            execution_started,
            timeout,
        }
    }
}

impl io::Write for BoundedRequestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.execution_started.elapsed() >= self.timeout {
            self.deadline_elapsed = true;
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "search request deadline exceeded",
            ));
        }
        let remaining = MAX_SEARCH_REQUEST_BYTES.saturating_sub(self.body.len());
        if buf.len() > remaining {
            self.exceeded = true;
            return Err(io::Error::other("search request body limit exceeded"));
        }
        self.body.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------

/// Translate the OpenAI Responses hybrid-search shape to OGX's typed schema.
fn translate_ranking_options(source: Option<&Value>) -> Result<TranslatedRankingOptions<'_>, &'static str> {
    let Some(source) = ranking_options_object(source)? else {
        return Ok(TranslatedRankingOptions {
            options: None,
            search_mode: None,
        });
    };

    let score_threshold = source.get("score_threshold");
    validate_ranker(source)?;
    match source.get("hybrid_search") {
        None => Ok(TranslatedRankingOptions {
            options: score_threshold.is_some().then_some(OgxSearchRankingOptions {
                ranker: None,
                alpha: None,
                score_threshold,
            }),
            search_mode: None,
        }),
        Some(hybrid) => translate_hybrid_ranking_options(hybrid, score_threshold),
    }
}

/// Map OpenAI's default selectors to OGX's configured default ranker.
fn validate_ranker(source: &serde_json::Map<String, Value>) -> Result<(), &'static str> {
    match source.get("ranker") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(ranker)) if matches!(ranker.as_str(), "auto" | "default-2024-11-15") => Ok(()),
        Some(Value::String(_ranker)) => Err("ranking_options.ranker is not supported"),
        Some(_) => Err("ranking_options.ranker must be a string"),
    }
}

/// Translate Responses hybrid weights to OGX's weighted ranker shape.
fn translate_hybrid_ranking_options<'a>(
    hybrid: &'a Value,
    score_threshold: Option<&'a Value>,
) -> Result<TranslatedRankingOptions<'a>, &'static str> {
    let hybrid = hybrid
        .as_object()
        .ok_or("ranking_options.hybrid_search must be an object")?;
    let alpha = normalized_hybrid_alpha(hybrid)?;

    Ok(TranslatedRankingOptions {
        options: Some(OgxSearchRankingOptions {
            ranker: Some("weighted"),
            alpha: Some(alpha),
            score_threshold,
        }),
        search_mode: Some("hybrid"),
    })
}

/// Borrow ranking options as an object while preserving an absent value.
fn ranking_options_object(source: Option<&Value>) -> Result<Option<&serde_json::Map<String, Value>>, &'static str> {
    source
        .map(|value| value.as_object().ok_or("ranking_options must be an object"))
        .transpose()
}

/// Normalize required Responses hybrid weights into OGX's semantic alpha.
fn normalized_hybrid_alpha(hybrid: &serde_json::Map<String, Value>) -> Result<f64, &'static str> {
    let embedding_weight = hybrid
        .get("embedding_weight")
        .and_then(Value::as_f64)
        .ok_or("ranking_options.hybrid_search.embedding_weight must be numeric")?;
    let text_weight = hybrid
        .get("text_weight")
        .and_then(Value::as_f64)
        .ok_or("ranking_options.hybrid_search.text_weight must be numeric")?;
    if !embedding_weight.is_finite() || !text_weight.is_finite() || embedding_weight < 0.0 || text_weight < 0.0 {
        return Err("ranking_options.hybrid_search weights must be finite and non-negative");
    }
    let total_weight = embedding_weight + text_weight;
    if total_weight == 0.0 || !total_weight.is_finite() {
        return Err("ranking_options.hybrid_search weights must have a finite positive sum");
    }
    let alpha = embedding_weight / total_weight;
    alpha
        .is_finite()
        .then_some(alpha)
        .ok_or("ranking_options.hybrid_search weights cannot be normalized")
}

/// Calculate the one-time aggregate admission needed by an execution.
fn response_admission_units(
    max_response_bytes: usize,
    max_total_response_bytes: usize,
    spec_count: usize,
) -> Result<u32, &'static str> {
    let aggregate_bytes = max_response_bytes
        .saturating_mul(spec_count)
        .min(max_total_response_bytes);
    let admitted_bytes = aggregate_bytes
        .checked_mul(RESPONSE_DECODE_MEMORY_MULTIPLIER)
        .ok_or("response admission is too large")?;
    u32::try_from(admitted_bytes.div_ceil(RESPONSE_BODY_BUDGET_UNIT_BYTES))
        .map_err(|_overflow| "response admission is too large")
}

/// Serialize one request without ever retaining an oversized body.
fn serialize_bounded_request(
    request: &VectorStoreSearchRequest<'_>,
    store_id: &str,
    execution_started: Instant,
    timeout: Duration,
) -> Result<Vec<u8>, FileSearchError> {
    let mut writer = BoundedRequestWriter::new(execution_started, timeout);
    let serialized = serde_json::to_writer(&mut writer, request);
    if writer.deadline_elapsed {
        return Err(execution_deadline_error(store_id));
    }
    if writer.exceeded {
        return Err(request_error(
            store_id,
            format!("search request exceeds {MAX_SEARCH_REQUEST_BYTES} byte limit"),
        ));
    }
    serialized.map_err(|error| request_error(store_id, format!("failed to serialize search request: {error}")))?;
    Ok(writer.body)
}

/// Build an outbound request error without copying an unbounded identifier.
pub(super) fn request_error(store_id: &str, message: impl Into<String>) -> FileSearchError {
    FileSearchError::Callout {
        message: message.into(),
        store_id: bounded_store_id(store_id),
    }
}

/// Keep error labels bounded even when the rejected identifier is oversized.
fn bounded_store_id(store_id: &str) -> String {
    if store_id.len() <= MAX_VECTOR_STORE_ID_BYTES {
        return store_id.to_owned();
    }
    let mut bounded: String = store_id.chars().take(128).collect();
    bounded.push_str("...");
    bounded
}

/// Build the shared filter-execution deadline error.
fn execution_deadline_error(store_id: &str) -> FileSearchError {
    request_error(
        store_id,
        "file-search execution deadline exceeded while sending or collecting a response",
    )
}

/// Return the remaining shared execution time or a bounded deadline error.
fn deadline_remaining(
    timeout: Duration,
    execution_started: Instant,
    store_id: &str,
) -> Result<Duration, FileSearchError> {
    timeout
        .checked_sub(execution_started.elapsed())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| execution_deadline_error(store_id))
}

/// Decode a collected body off the Tokio worker under the shared deadline.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps one response's deadline and admission ownership explicit"
)]
async fn parse_response_body_with_deadline(
    body: Vec<u8>,
    store_id: &str,
    result_limit: usize,
    execution_started: Instant,
    timeout: Duration,
    response_admission: Arc<ResponseAdmission>,
) -> Result<SearchResponse, FileSearchError> {
    let remaining = deadline_remaining(timeout, execution_started, store_id)?;
    let decode_slot = tokio::time::timeout(remaining, RESPONSE_DECODE_SLOTS.acquire())
        .await
        .map_err(|_elapsed| execution_deadline_error(store_id))?
        .map_err(|_closed| request_error(store_id, "response decoder admission is unavailable"))?;
    let remaining = deadline_remaining(timeout, execution_started, store_id)?;
    let error_store_id = bounded_store_id(store_id);
    let parse_store_id = error_store_id.clone();
    let decode_deadline = execution_started.checked_add(timeout).unwrap_or(execution_started);
    let mut task = tokio::task::spawn_blocking(move || {
        let _response_admission = response_admission;
        let _decode_slot = decode_slot;
        parse_response_body(&body, &parse_store_id, result_limit, decode_deadline)
    });
    match tokio::time::timeout(remaining, &mut task).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => Err(request_error(
            &error_store_id,
            format!("response decoding task failed: {error}"),
        )),
        Err(_elapsed) => {
            task.abort();
            Err(execution_deadline_error(&error_store_id))
        },
    }
}

/// Parse one bounded API response without retaining its response buffer.
fn parse_response_body(
    body: &[u8],
    store_id: &str,
    result_limit: usize,
    decode_deadline: Instant,
) -> Result<SearchResponse, FileSearchError> {
    let body_bytes = body.len();
    let data = deserialize_bounded_search_results(body, result_limit, decode_deadline).map_err(|error| {
        if Instant::now() >= decode_deadline {
            return execution_deadline_error(store_id);
        }
        FileSearchError::Deserialize {
            body_bytes,
            line: error.line(),
            column: error.column(),
            store_id: store_id.to_owned(),
        }
    })?;
    Ok(SearchResponse { body_bytes, data })
}

/// Deserialize the response page while retaining only its top-k results.
fn deserialize_bounded_search_results(
    body: &[u8],
    result_limit: usize,
    decode_deadline: Instant,
) -> Result<Vec<SearchResult>, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let results = SearchResponseSeed {
        decode_deadline,
        result_limit,
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(results)
}

/// Seed that rejects an oversized map key before retaining its owned copy.
struct BoundedMapKeySeed {
    /// Maximum accepted UTF-8 bytes.
    max_bytes: usize,

    /// Static rejection text that cannot include attacker-controlled input.
    limit_error: &'static str,
}

impl<'de> DeserializeSeed<'de> for BoundedMapKeySeed {
    type Value = String;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_identifier(BoundedMapKeyVisitor {
            max_bytes: self.max_bytes,
            limit_error: self.limit_error,
        })
    }
}

/// Visitor that owns only keys already proven to fit their schema limit.
struct BoundedMapKeyVisitor {
    /// Maximum accepted UTF-8 bytes.
    max_bytes: usize,

    /// Static rejection text that cannot include attacker-controlled input.
    limit_error: &'static str,
}

impl Visitor<'_> for BoundedMapKeyVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded vector-store response field name")
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        if v.len() > self.max_bytes {
            return Err(E::custom(self.limit_error));
        }
        Ok(v.to_owned())
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        if v.len() > self.max_bytes {
            return Err(E::custom(self.limit_error));
        }
        Ok(v)
    }
}

/// Seed that validates and discards an extension key without copying it.
struct BoundedIgnoredMapKeySeed;

impl<'de> DeserializeSeed<'de> for BoundedIgnoredMapKeySeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_identifier(BoundedIgnoredMapKeyVisitor)
    }
}

/// Visitor for ignored extension keys.
struct BoundedIgnoredMapKeyVisitor;

impl Visitor<'_> for BoundedIgnoredMapKeyVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded vector-store extension field name")
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        if v.len() > MAX_IGNORED_EXTENSION_STRING_BYTES {
            return Err(E::custom("vector-store content extension key exceeds the byte limit"));
        }
        Ok(())
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        self.visit_str(&v)
    }
}

/// Seed for one bounded vector-store response page.
struct SearchResponseSeed {
    /// Hard deadline checked during response traversal.
    decode_deadline: Instant,

    /// Maximum results retained from the page.
    result_limit: usize,
}

impl<'de> DeserializeSeed<'de> for SearchResponseSeed {
    type Value = Vec<SearchResult>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(SearchResponseVisitor {
            decode_deadline: self.decode_deadline,
            result_limit: self.result_limit,
        })
    }
}

/// Visitor that validates bounded page metadata and requires one `data` array.
struct SearchResponseVisitor {
    /// Hard deadline checked during response traversal.
    decode_deadline: Instant,

    /// Maximum results retained from the page.
    result_limit: usize,
}

impl<'de> Visitor<'de> for SearchResponseVisitor {
    type Value = Vec<SearchResult>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store search response object")
    }

    #[expect(clippy::too_many_lines, reason = "validates the complete bounded page schema")]
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut results = None;
        while let Some(key) = map.next_key_seed(BoundedMapKeySeed {
            max_bytes: MAX_SCHEMA_FIELD_BYTES,
            limit_error: "vector-store response field name exceeds the byte limit",
        })? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            match key.as_str() {
                "data" => {
                    if results.is_some() {
                        return Err(A::Error::duplicate_field("data"));
                    }
                    results = Some(map.next_value_seed(SearchResultsSeed {
                        decode_deadline: self.decode_deadline,
                        result_limit: self.result_limit,
                    })?);
                },
                "object" => {
                    let object = map.next_value::<String>()?;
                    if object != "vector_store.search_results.page" {
                        return Err(A::Error::custom("invalid vector-store response object type"));
                    }
                },
                "search_query" => {
                    map.next_value_seed(BoundedStringListSeed {
                        decode_deadline: self.decode_deadline,
                        max_items: MAX_QUERIES_PER_RESPONSE,
                        max_string_bytes: MAX_QUERY_BYTES,
                    })?;
                },
                "has_more" => {
                    map.next_value::<bool>()?;
                },
                "next_page" => {
                    let next_page = map.next_value::<Option<String>>()?;
                    if next_page
                        .as_ref()
                        .is_some_and(|value| value.len() > MAX_NEXT_PAGE_BYTES)
                    {
                        return Err(A::Error::custom("vector-store next-page token exceeds the byte limit"));
                    }
                },
                _ => {
                    return Err(A::Error::custom("unknown vector-store response field"));
                },
            }
        }
        results.ok_or_else(|| A::Error::missing_field("data"))
    }
}

/// Seed for the bounded `data` result array.
struct SearchResultsSeed {
    /// Hard deadline checked while decoding result entries.
    decode_deadline: Instant,

    /// Maximum results retained from the array.
    result_limit: usize,
}

impl<'de> DeserializeSeed<'de> for SearchResultsSeed {
    type Value = Vec<SearchResult>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(SearchResultsVisitor {
            decode_deadline: self.decode_deadline,
            result_limit: self.result_limit,
        })
    }
}

/// Seed for one result whose nested content is deadline and count bounded.
struct SearchResultSeed {
    /// Hard deadline checked during nested traversal.
    decode_deadline: Instant,
}

impl<'de> DeserializeSeed<'de> for SearchResultSeed {
    type Value = SearchResult;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(SearchResultVisitor {
            decode_deadline: self.decode_deadline,
        })
    }
}

/// Visitor for the exact OGX result fields consumed by the filter.
struct SearchResultVisitor {
    /// Hard deadline checked between fields.
    decode_deadline: Instant,
}

impl<'de> Visitor<'de> for SearchResultVisitor {
    type Value = SearchResult;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store search result")
    }

    #[expect(clippy::too_many_lines, reason = "owns and validates every required result field")]
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut attributes = None;
        let mut attributes_seen = false;
        let mut content = None;
        let mut file_id = None;
        let mut filename = None;
        let mut score = None;
        while let Some(key) = map.next_key_seed(BoundedMapKeySeed {
            max_bytes: MAX_SCHEMA_FIELD_BYTES,
            limit_error: "vector-store result field name exceeds the byte limit",
        })? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            match key.as_str() {
                "attributes" => {
                    if attributes_seen {
                        return Err(A::Error::duplicate_field("attributes"));
                    }
                    attributes_seen = true;
                    attributes = map.next_value_seed(OptionalAttributesSeed {
                        decode_deadline: self.decode_deadline,
                    })?;
                },
                "content" => {
                    if content.is_some() {
                        return Err(A::Error::duplicate_field("content"));
                    }
                    content = Some(map.next_value_seed(ContentChunksSeed {
                        decode_deadline: self.decode_deadline,
                    })?);
                },
                "file_id" => set_once(&mut file_id, map.next_value()?, "file_id")?,
                "filename" => set_once(&mut filename, map.next_value()?, "filename")?,
                "score" => set_once(&mut score, map.next_value()?, "score")?,
                _ => {
                    return Err(A::Error::custom("unknown vector-store result field"));
                },
            }
        }
        let file_id = canonical_ogx_file_id(&attributes, file_id.ok_or_else(|| A::Error::missing_field("file_id"))?);
        let result = SearchResult {
            attributes,
            content: content.ok_or_else(|| A::Error::missing_field("content"))?,
            file_id,
            filename: filename.ok_or_else(|| A::Error::missing_field("filename"))?,
            score: score.ok_or_else(|| A::Error::missing_field("score"))?,
        };
        Ok(result)
    }
}

/// Select the OpenAI Files API ID from OGX's indexed-document result shape.
///
/// OGX may return its internal document UUID as the top-level `file_id` and
/// retain the source `file-*` identifier in attributes. Canonical responses
/// already carrying a `file-*` ID remain authoritative.
fn canonical_ogx_file_id(attributes: &Option<Value>, file_id: String) -> String {
    if file_id.strip_prefix("file-").is_some_and(|suffix| !suffix.is_empty()) {
        return file_id;
    }
    attributes
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|attributes| attributes.get("file_id"))
        .and_then(Value::as_str)
        .filter(|candidate| candidate.strip_prefix("file-").is_some_and(|suffix| !suffix.is_empty()))
        .map_or(file_id, str::to_owned)
}

/// Seed for nullable, OpenAI-compatible result attributes.
struct OptionalAttributesSeed {
    /// Hard deadline checked between attribute entries.
    decode_deadline: Instant,
}

impl<'de> DeserializeSeed<'de> for OptionalAttributesSeed {
    type Value = Option<Value>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_option(OptionalAttributesVisitor {
            decode_deadline: self.decode_deadline,
        })
    }
}

/// Visitor for a nullable bounded attributes object.
struct OptionalAttributesVisitor {
    /// Hard deadline checked between attribute entries.
    decode_deadline: Instant,
}

impl<'de> Visitor<'de> for OptionalAttributesVisitor {
    type Value = Option<Value>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("null or an OpenAI-compatible attributes object")
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer
            .deserialize_map(AttributesVisitor {
                decode_deadline: self.decode_deadline,
            })
            .map(Some)
    }
}

/// Visitor retaining at most the public Responses attributes schema.
struct AttributesVisitor {
    /// Hard deadline checked between attribute entries.
    decode_deadline: Instant,
}

impl<'de> Visitor<'de> for AttributesVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an OpenAI-compatible attributes object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut attributes = serde_json::Map::new();
        while let Some(key) = map.next_key_seed(BoundedMapKeySeed {
            max_bytes: MAX_ATTRIBUTE_KEY_CHARS.saturating_mul(4),
            limit_error: "vector-store result attribute key exceeds the byte limit",
        })? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            if attributes.len() == MAX_ATTRIBUTE_ENTRIES {
                return Err(A::Error::custom(
                    "vector-store result attributes exceed the entry limit",
                ));
            }
            if exceeds_character_limit(&key, MAX_ATTRIBUTE_KEY_CHARS) {
                return Err(A::Error::custom(
                    "vector-store result attribute key exceeds the character limit",
                ));
            }
            let value = map.next_value_seed(AttributeValueSeed)?;
            if attributes.insert(key, value).is_some() {
                return Err(A::Error::custom("duplicate vector-store result attribute"));
            }
        }
        Ok(Value::Object(attributes))
    }
}

/// Seed for one primitive public attribute value.
struct AttributeValueSeed;

impl<'de> DeserializeSeed<'de> for AttributeValueSeed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(AttributeValueVisitor)
    }
}

/// Visitor rejecting nested or oversized public attribute values.
struct AttributeValueVisitor;

impl Visitor<'_> for AttributeValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded string, number, or boolean attribute")
    }

    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
        serde_json::Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("vector-store result attribute number must be finite"))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        if exceeds_character_limit(v, MAX_ATTRIBUTE_STRING_CHARS) {
            return Err(E::custom(
                "vector-store result string attribute exceeds the character limit",
            ));
        }
        Ok(Value::String(v.to_owned()))
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        if exceeds_character_limit(&v, MAX_ATTRIBUTE_STRING_CHARS) {
            return Err(E::custom(
                "vector-store result string attribute exceeds the character limit",
            ));
        }
        Ok(Value::String(v))
    }
}

/// Store one required map field or reject a duplicate.
fn set_once<T, E: serde::de::Error>(slot: &mut Option<T>, value: T, field: &'static str) -> Result<(), E> {
    if slot.replace(value).is_some() {
        return Err(E::duplicate_field(field));
    }
    Ok(())
}

/// Seed for bounded content chunks within one result.
struct ContentChunksSeed {
    /// Hard deadline checked between chunks.
    decode_deadline: Instant,
}

impl<'de> DeserializeSeed<'de> for ContentChunksSeed {
    type Value = Vec<ContentChunk>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(ContentChunksVisitor {
            decode_deadline: self.decode_deadline,
        })
    }
}

/// Visitor that rejects excess chunks before retaining more work.
struct ContentChunksVisitor {
    /// Hard deadline checked between chunks.
    decode_deadline: Instant,
}

impl<'de> Visitor<'de> for ContentChunksVisitor {
    type Value = Vec<ContentChunk>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded vector-store content array")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut chunks = Vec::new();
        while let Some(chunk) = seq.next_element_seed(ContentChunkSeed {
            decode_deadline: self.decode_deadline,
        })? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            if chunks.len() == MAX_DECODED_CONTENT_CHUNKS {
                return Err(A::Error::custom("vector-store result exceeds the content chunk limit"));
            }
            chunks.push(chunk);
        }
        Ok(chunks)
    }
}

/// Seed for one exact text content chunk.
struct ContentChunkSeed {
    /// Hard deadline checked between fields.
    decode_deadline: Instant,
}

impl<'de> DeserializeSeed<'de> for ContentChunkSeed {
    type Value = ContentChunk;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(ContentChunkVisitor {
            decode_deadline: self.decode_deadline,
        })
    }
}

/// Visitor for the text chunk subset exposed by the OpenAI search endpoint.
struct ContentChunkVisitor {
    /// Hard deadline checked between fields.
    decode_deadline: Instant,
}

impl<'de> Visitor<'de> for ContentChunkVisitor {
    type Value = ContentChunk;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store text content chunk")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut chunk_type = None;
        let mut text = None;
        while let Some(key) = map.next_key_seed(BoundedMapKeySeed {
            max_bytes: MAX_SCHEMA_FIELD_BYTES,
            limit_error: "vector-store content field name exceeds the byte limit",
        })? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            match key.as_str() {
                "type" => set_once(&mut chunk_type, map.next_value()?, "type")?,
                "text" => set_once(&mut text, map.next_value()?, "text")?,
                "embedding" | "chunk_metadata" | "metadata" => {
                    map.next_value_seed(BoundedIgnoredSeed {
                        decode_deadline: self.decode_deadline,
                        depth: 0,
                    })?;
                },
                _ => {
                    return Err(A::Error::custom("unknown vector-store content field"));
                },
            }
        }
        let text: String = text.ok_or_else(|| A::Error::missing_field("text"))?;
        if text.len() > MAX_CONTENT_CHUNK_BYTES {
            return Err(A::Error::custom("vector-store content chunk exceeds the byte limit"));
        }
        Ok(ContentChunk {
            _chunk_type: chunk_type.ok_or_else(|| A::Error::missing_field("type"))?,
            text,
        })
    }
}

/// Seed traversing an unused extension with deadline and shape bounds.
struct BoundedIgnoredSeed {
    /// Hard deadline checked throughout traversal.
    decode_deadline: Instant,

    /// Current nested container depth.
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedIgnoredSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(BoundedIgnoredVisitor {
            decode_deadline: self.decode_deadline,
            depth: self.depth,
        })
    }
}

/// Visitor discarding a bounded extension without retaining its values.
struct BoundedIgnoredVisitor {
    /// Hard deadline checked throughout traversal.
    decode_deadline: Instant,

    /// Current nested container depth.
    depth: usize,
}

impl<'de> Visitor<'de> for BoundedIgnoredVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded JSON extension value")
    }

    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_bool<E: serde::de::Error>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E: serde::de::Error>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E: serde::de::Error>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E: serde::de::Error>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        ensure_decode_deadline::<E>(self.decode_deadline)?;
        if v.len() > MAX_IGNORED_EXTENSION_STRING_BYTES {
            return Err(E::custom(
                "vector-store content extension string exceeds the byte limit",
            ));
        }
        Ok(())
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        self.visit_str(&v)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        BoundedIgnoredSeed {
            decode_deadline: self.decode_deadline,
            depth: self.depth,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if self.depth >= MAX_IGNORED_EXTENSION_DEPTH {
            return Err(A::Error::custom(
                "vector-store content extension exceeds the nesting limit",
            ));
        }
        let mut entries = 0_usize;
        while seq
            .next_element_seed(BoundedIgnoredSeed {
                decode_deadline: self.decode_deadline,
                depth: self.depth.saturating_add(1),
            })?
            .is_some()
        {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            entries = entries.saturating_add(1);
            if entries > MAX_IGNORED_EXTENSION_ENTRIES {
                return Err(A::Error::custom(
                    "vector-store content extension exceeds the entry limit",
                ));
            }
        }
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        if self.depth >= MAX_IGNORED_EXTENSION_DEPTH {
            return Err(A::Error::custom(
                "vector-store content extension exceeds the nesting limit",
            ));
        }
        let mut entries = 0_usize;
        while map.next_key_seed(BoundedIgnoredMapKeySeed)?.is_some() {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            entries = entries.saturating_add(1);
            if entries > MAX_IGNORED_EXTENSION_ENTRIES {
                return Err(A::Error::custom(
                    "vector-store content extension exceeds the entry limit",
                ));
            }
            map.next_value_seed(BoundedIgnoredSeed {
                decode_deadline: self.decode_deadline,
                depth: self.depth.saturating_add(1),
            })?;
        }
        Ok(())
    }
}

/// Seed for bounded search-query metadata ignored by aggregation.
struct BoundedStringListSeed {
    /// Hard deadline checked between list entries.
    decode_deadline: Instant,

    /// Maximum strings accepted in the list.
    max_items: usize,

    /// Maximum bytes accepted in one string.
    max_string_bytes: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedStringListSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(BoundedStringListVisitor {
            decode_deadline: self.decode_deadline,
            max_items: self.max_items,
            max_string_bytes: self.max_string_bytes,
        })
    }
}

/// Visitor that validates ignored string-list metadata without retaining it.
struct BoundedStringListVisitor {
    /// Hard deadline checked between list entries.
    decode_deadline: Instant,

    /// Maximum strings accepted in the list.
    max_items: usize,

    /// Maximum bytes accepted in one string.
    max_string_bytes: usize,
}

impl<'de> Visitor<'de> for BoundedStringListVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded list of strings")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut count = 0_usize;
        while let Some(value) = seq.next_element::<String>()? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            count = count.saturating_add(1);
            if count > self.max_items || value.len() > self.max_string_bytes {
                return Err(A::Error::custom("vector-store search-query metadata exceeds its limit"));
            }
        }
        Ok(())
    }
}

/// Streaming top-k visitor for vector-store results.
struct SearchResultsVisitor {
    /// Hard deadline checked while decoding result entries.
    decode_deadline: Instant,

    /// Maximum results retained from the array.
    result_limit: usize,
}

impl<'de> Visitor<'de> for SearchResultsVisitor {
    type Value = Vec<SearchResult>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array of vector-store search results")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut decoded_results = 0_usize;
        let mut decoded_chunks = 0_usize;
        let mut results = Vec::with_capacity(self.result_limit);
        while let Some(result) = seq.next_element_seed(SearchResultSeed {
            decode_deadline: self.decode_deadline,
        })? {
            ensure_decode_deadline::<A::Error>(self.decode_deadline)?;
            decoded_results = decoded_results.saturating_add(1);
            if decoded_results > MAX_NUM_RESULTS {
                return Err(A::Error::custom("vector-store response exceeds the result count limit"));
            }
            decoded_chunks = decoded_chunks
                .checked_add(result.content.len())
                .ok_or_else(|| A::Error::custom("vector-store content chunk count overflow"))?;
            if decoded_chunks > MAX_DECODED_CONTENT_CHUNKS {
                return Err(A::Error::custom(
                    "vector-store response exceeds the content chunk count limit",
                ));
            }
            merge_top_results(&mut results, std::iter::once(result), self.result_limit);
        }
        Ok(results)
    }
}

/// Reject a decoder that has crossed its shared execution deadline.
fn ensure_decode_deadline<E: serde::de::Error>(deadline: Instant) -> Result<(), E> {
    if Instant::now() >= deadline {
        return Err(E::custom("vector-store response decode deadline exceeded"));
    }
    Ok(())
}

/// Return whether a string contains more than the permitted characters.
fn exceeds_character_limit(value: &str, limit: usize) -> bool {
    value.chars().nth(limit).is_some()
}

/// Merge one bounded concurrency chunk into the aggregate batch.
fn merge_chunk_results(
    batch: &mut SearchBatch,
    consumed_bytes: &mut usize,
    specs: &[SearchSpec<'_>],
    results: Vec<Result<SearchResponse, FileSearchError>>,
    total_limit: usize,
) -> bool {
    let mut failed = false;
    for (spec, result) in specs.iter().zip(results) {
        let result = merge_search_result(batch, consumed_bytes, spec, result, total_limit);
        if let Err(error) = result {
            failed = true;
            batch.failures.push(SearchFailure {
                call_index: spec.call_index,
                error,
            });
        }
    }
    failed
}

/// Account for and retain only the top results from one response.
fn merge_search_result(
    batch: &mut SearchBatch,
    consumed_bytes: &mut usize,
    spec: &SearchSpec<'_>,
    response: Result<SearchResponse, FileSearchError>,
    total_limit: usize,
) -> Result<(), FileSearchError> {
    let body_bytes = match &response {
        Ok(response) => response.body_bytes,
        Err(FileSearchError::Deserialize { body_bytes, .. }) => *body_bytes,
        Err(FileSearchError::Callout { .. }) => 0,
    };
    let total = consumed_bytes
        .checked_add(body_bytes)
        .filter(|total| *total <= total_limit)
        .ok_or_else(|| aggregate_limit_error(spec.store_id, total_limit))?;
    *consumed_bytes = total;

    let SearchResponse { body_bytes: _, data } = response?;
    {
        let results = batch
            .results_by_call
            .get_mut(spec.call_index)
            .ok_or_else(|| FileSearchError::Callout {
                message: "search spec references an unknown call index".to_owned(),
                store_id: spec.store_id.to_owned(),
            })?;
        merge_top_results(results, data, result_limit(spec.max_num_results));
    }
    Ok(())
}

/// Record a pre-fan-out admission failure for every planned search.
fn append_admission_failures(failures: &mut Vec<SearchFailure>, specs: &[SearchSpec<'_>], message: &'static str) {
    failures.extend(specs.iter().map(|spec| SearchFailure {
        call_index: spec.call_index,
        error: request_error(spec.store_id, message),
    }));
}

/// Append one aggregate-budget failure for each unexecuted spec.
fn append_budget_failures(failures: &mut Vec<SearchFailure>, specs: &[SearchSpec<'_>], limit: usize) {
    failures.extend(specs.iter().map(|spec| SearchFailure {
        call_index: spec.call_index,
        error: aggregate_limit_error(spec.store_id, limit),
    }));
}

/// Mark specs skipped because the shared execution deadline has elapsed.
fn append_deadline_failures(failures: &mut Vec<SearchFailure>, specs: &[SearchSpec<'_>]) {
    failures.extend(specs.iter().map(|spec| SearchFailure {
        call_index: spec.call_index,
        error: execution_deadline_error(spec.store_id),
    }));
}

/// Mark specs deliberately not scheduled after a fail-closed chunk failed.
fn append_fail_closed_failures(failures: &mut Vec<SearchFailure>, specs: &[SearchSpec<'_>]) {
    failures.extend(specs.iter().map(|spec| SearchFailure {
        call_index: spec.call_index,
        error: request_error(
            spec.store_id,
            "search not scheduled after an earlier fail-closed callout failed",
        ),
    }));
}

/// Build an aggregate-budget error.
fn aggregate_limit_error(store_id: &str, limit: usize) -> FileSearchError {
    request_error(
        store_id,
        format!("aggregate response body limit of {limit} bytes reached"),
    )
}

/// Merge response results without retaining more than the final top-k.
fn merge_top_results(target: &mut Vec<SearchResult>, incoming: impl IntoIterator<Item = SearchResult>, limit: usize) {
    if limit == 0 {
        return;
    }

    for candidate in incoming {
        if target.len() < limit {
            target.push(candidate);
            continue;
        }

        let Some((lowest_index, lowest)) = target
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.score.total_cmp(&right.score))
        else {
            continue;
        };
        if candidate.score.total_cmp(&lowest.score).is_gt()
            && let Some(slot) = target.get_mut(lowest_index)
        {
            *slot = candidate;
        }
    }
}

/// Resolve the bounded number of results retained for a call.
fn result_limit(configured: Option<u64>) -> usize {
    match configured {
        None => DEFAULT_MAX_NUM_RESULTS,
        Some(value) => usize::try_from(value).unwrap_or(MAX_NUM_RESULTS).min(MAX_NUM_RESULTS),
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use explicit construction assertions")]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    };

    use serde_json::json;

    use super::{
        MAX_ATTRIBUTE_ENTRIES, MAX_DECODED_CONTENT_CHUNKS, SearchResult, VectorStoreSearchRequest,
        deserialize_bounded_search_results, merge_top_results, parse_response_body, response_admission_units,
    };

    fn decode(body: &[u8], limit: usize) -> Result<Vec<SearchResult>, serde_json::Error> {
        deserialize_bounded_search_results(body, limit, Instant::now() + Duration::from_secs(60))
    }

    fn result(file_id: &str, score: f64) -> SearchResult {
        serde_json::from_value(json!({
            "attributes": null,
            "content": [{"type":"text","text":file_id}],
            "file_id": file_id,
            "filename": format!("{file_id}.txt"),
            "score": score,
        }))
        .expect("test result must deserialize")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn aggregate_admission_serializes_whole_many_spec_executions() {
        let units = response_admission_units(10_485_760, 67_108_864, 64).expect("admission must fit");
        assert_eq!(units, 128);
        let budget = Arc::new(tokio::sync::Semaphore::new(
            usize::try_from(units).expect("u32 units must fit usize") * 2,
        ));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let tasks = (0..3).map(|_| {
            let budget = Arc::clone(&budget);
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            tokio::spawn(async move {
                let _permit = budget
                    .acquire_many_owned(units)
                    .await
                    .expect("test semaphore remains open");
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                active.fetch_sub(1, Ordering::SeqCst);
            })
        });

        for result in futures::future::join_all(tasks).await {
            result.expect("admission task must complete");
        }
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn bounded_decoder_retains_top_ranked_chunks() {
        let body = serde_json::to_vec(&json!({
            "data": [
                {"content":[{"type":"text","text":"old"}],"file_id":"file-a","filename":"a.txt","score":0.2},
                {"content":[{"type":"text","text":"b"}],"file_id":"file-b","filename":"b.txt","score":0.8},
                {"content":[{"type":"text","text":"new"}],"file_id":"file-a","filename":"a.txt","score":0.9},
                {"content":[{"type":"text","text":"c"}],"file_id":"file-c","filename":"c.txt","score":0.7}
            ]
        }))
        .expect("test response must serialize");

        let decoded = decode(&body, 4).expect("response must decode");

        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded.iter().filter(|result| result.file_id == "file-a").count(), 2);
        assert!(
            decoded
                .iter()
                .any(|result| result.file_id == "file-a" && result.score == 0.9)
        );
        assert!(decoded.iter().any(|result| result.file_id == "file-b"));
    }

    #[test]
    fn aggregate_merge_preserves_distinct_chunks_from_one_file() {
        let mut aggregate = vec![result("file-a", 0.4), result("file-b", 0.6)];

        merge_top_results(&mut aggregate, vec![result("file-a", 0.9), result("file-c", 0.8)], 4);

        assert_eq!(aggregate.len(), 4);
        assert_eq!(aggregate.iter().filter(|result| result.file_id == "file-a").count(), 2);
        assert!(
            aggregate
                .iter()
                .any(|result| result.file_id == "file-a" && result.score == 0.9)
        );
        assert!(aggregate.iter().any(|result| result.file_id == "file-c"));
    }

    #[test]
    fn openai_default_rankers_use_ogx_configured_default() {
        for ranker in ["auto", "default-2024-11-15"] {
            let ranking_options = json!({"ranker": ranker, "score_threshold": 0.2});
            let request = VectorStoreSearchRequest::new(None, None, "query", Some(&ranking_options))
                .expect("ranking options must translate");
            let encoded = serde_json::to_value(request).expect("request must serialize");

            assert!(encoded.pointer("/ranking_options/ranker").is_none());
            assert_eq!(encoded.pointer("/ranking_options/score_threshold"), Some(&json!(0.2)));
            assert!(encoded.get("search_mode").is_none());
        }

        for ranker in ["default-attacker", "default-2025-01-01", "weighted"] {
            let ranking_options = json!({"ranker": ranker});
            assert!(
                VectorStoreSearchRequest::new(None, None, "query", Some(&ranking_options)).is_err(),
                "noncanonical ranker {ranker:?} must not be silently discarded"
            );
        }
    }

    #[test]
    fn bounded_decoder_requires_data_array() {
        assert!(decode(b"{}", 10).is_err());
        assert!(decode(br#"{"data":null}"#, 10).is_err());
    }

    #[test]
    fn bounded_decoder_rejects_excess_content_work() {
        let chunks = (0..=MAX_DECODED_CONTENT_CHUNKS)
            .map(|_| json!({"type":"text","text":"x"}))
            .collect::<Vec<_>>();
        let body = serde_json::to_vec(&json!({
            "data":[{"content":chunks,"file_id":"file-a","filename":"a.txt","score":0.9}]
        }))
        .expect("test response must serialize");

        assert!(decode(&body, 10).is_err());
    }

    #[test]
    fn bounded_decoder_rejects_nonconformant_attributes() {
        let nested = serde_json::to_vec(&json!({
            "data":[{
                "attributes":{"nested":{"value":true}},
                "content":[{"type":"text","text":"x"}],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }]
        }))
        .expect("test response must serialize");
        let too_many = serde_json::to_vec(&json!({
            "data":[{
                "attributes":(0..=MAX_ATTRIBUTE_ENTRIES)
                    .map(|index| (format!("key-{index}"), json!(index)))
                    .collect::<serde_json::Map<_,_>>(),
                "content":[{"type":"text","text":"x"}],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }]
        }))
        .expect("test response must serialize");
        let oversized = serde_json::to_vec(&json!({
            "data":[{
                "attributes":{"key":"x".repeat(super::MAX_ATTRIBUTE_STRING_CHARS + 1)},
                "content":[{"type":"text","text":"x"}],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }]
        }))
        .expect("test response must serialize");

        assert!(decode(&nested, 10).is_err());
        assert!(decode(&too_many, 10).is_err());
        assert!(decode(&oversized, 10).is_err());
    }

    #[test]
    fn bounded_decoder_accepts_primitive_attribute_values() {
        let decoded = decode(
            br#"{"data":[{
                "attributes":{"boolean":true,"negative":-1,"positive":1,"float":1.5,"string":"value"},
                "content":[{"type":"text","text":"result"}],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }]}"#,
            10,
        )
        .expect("primitive attributes must decode");
        let attributes = decoded
            .first()
            .and_then(|result| result.attributes.as_ref())
            .expect("attributes must be retained");

        assert_eq!(attributes.get("boolean"), Some(&json!(true)));
        assert_eq!(attributes.get("negative"), Some(&json!(-1)));
        assert_eq!(attributes.get("positive"), Some(&json!(1)));
        assert_eq!(attributes.get("float"), Some(&json!(1.5)));
        assert_eq!(attributes.get("string"), Some(&json!("value")));
    }

    #[test]
    fn bounded_decoder_rejects_invalid_container_shapes() {
        let cases: &[&[u8]] = &[
            b"[]",
            br#"{"data":{}}"#,
            br#"{"data":[1]}"#,
            br#"{"data":[{"content":1}]}"#,
            br#"{"data":[{"content":[1]}]}"#,
            br#"{"data":[{"content":[{"type":1,"text":"x"}]}]}"#,
            br#"{"data":[{"attributes":[]}]}"#,
            br#"{"search_query":"query","data":[]}"#,
        ];

        for body in cases {
            assert!(decode(body, 10).is_err(), "invalid shape must be rejected");
        }
    }

    #[test]
    fn bounded_decoder_rejects_duplicate_and_bounded_fields() {
        let cases: &[&[u8]] = &[
            br#"{"data":[],"data":[]}"#,
            br#"{"object":"wrong","data":[]}"#,
            br#"{"data":[{"attributes":null,"attributes":null}]}"#,
            br#"{"data":[{"content":[],"content":[]}]}"#,
            br#"{"data":[{"unknown":null}]}"#,
            br#"{"data":[{"file_id":"file-a","file_id":"file-b"}]}"#,
        ];
        for body in cases {
            assert!(decode(body, 10).is_err(), "invalid field must be rejected");
        }

        let oversized_page = serde_json::to_vec(&json!({
            "next_page":"x".repeat(super::MAX_NEXT_PAGE_BYTES + 1),
            "data":[],
        }))
        .expect("test response must serialize");
        assert!(decode(&oversized_page, 10).is_err());

        let oversized_attribute_key = "x".repeat(super::MAX_ATTRIBUTE_KEY_CHARS + 1);
        let oversized_attribute = serde_json::to_vec(&json!({
            "data":[{
                "attributes":{oversized_attribute_key:null},
                "content":[{"type":"text","text":"x"}],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }]
        }))
        .expect("test response must serialize");
        assert!(decode(&oversized_attribute, 10).is_err());
    }

    #[test]
    fn bounded_decoder_uses_ogx_source_file_id_for_internal_documents() {
        let decoded = decode(
            br#"{"data":[
                {
                    "attributes":{"file_id":"file-source"},
                    "content":[{"type":"text","text":"fallback"}],
                    "file_id":"83441278-02e0-44bb-b385-892a1d4680c5",
                    "filename":"report.txt",
                    "score":0.9
                },
                {
                    "attributes":{"file_id":"file-shadow"},
                    "content":[{"type":"text","text":"canonical"}],
                    "file_id":"file-canonical",
                    "filename":"report.txt",
                    "score":0.8
                }
            ]}"#,
            10,
        )
        .expect("OGX result IDs must decode");

        let mut decoded = decoded.into_iter();
        assert_eq!(
            decoded.next().map(|result| result.file_id).as_deref(),
            Some("file-source")
        );
        assert_eq!(
            decoded.next().map(|result| result.file_id).as_deref(),
            Some("file-canonical")
        );
    }

    #[test]
    fn bounded_decoder_accepts_bounded_ogx_content_extensions() {
        let body = serde_json::to_vec(&json!({
            "object":"vector_store.search_results.page",
            "search_query":["query"],
            "data":[{
                "attributes":{"team":"infra"},
                "content":[{
                    "type":"text",
                    "text":"result",
                    "embedding":[0.1, 0.2],
                    "chunk_metadata":{"index":1},
                    "metadata":{"tags":["a", "b"]}
                }],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }],
            "has_more":false,
            "next_page":null
        }))
        .expect("test response must serialize");

        let decoded = decode(&body, 10).expect("bounded OGX extensions must decode");
        assert_eq!(decoded.len(), 1);
    }

    #[test]
    fn bounded_decoder_checks_deadline_during_nested_traversal() {
        let body = br#"{"data":[{"content":[{"type":"text","text":"x","metadata":{"nested":[1,2,3]}}],"file_id":"file-a","filename":"a.txt","score":0.9}]}"#;
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        assert!(deserialize_bounded_search_results(body, 10, expired).is_err());
    }

    #[test]
    fn bounded_decoder_rejects_unknown_top_level_payloads_without_traversal() {
        let body = br#"{"unknown":{"large":[1,2,3]},"data":[]}"#;

        assert!(decode(body, 10).is_err());
    }

    #[test]
    fn bounded_decoder_does_not_echo_oversized_map_keys_or_content_types() {
        let oversized_key = format!("sensitive-{}", "x".repeat(1_024));
        let unknown = serde_json::to_vec(&json!({oversized_key.clone(): null, "data": []}))
            .expect("test response must serialize");
        let unknown_error = decode(&unknown, 10).expect_err("oversized key must fail").to_string();
        assert!(unknown_error.len() < 256);
        assert!(!unknown_error.contains(&oversized_key));

        let oversized_type = format!("sensitive-{}", "y".repeat(1_024));
        let content = serde_json::to_vec(&json!({
            "data":[{
                "content":[{"type":oversized_type.clone(),"text":"x"}],
                "file_id":"file-a","filename":"a.txt","score":0.9
            }]
        }))
        .expect("test response must serialize");
        let content_error = decode(&content, 10)
            .expect_err("unsupported type must fail")
            .to_string();
        assert!(content_error.len() < 256);
        assert!(!content_error.contains(&oversized_type));
    }

    #[test]
    fn production_decode_error_retains_only_bounded_location_metadata() {
        let secret = format!("secret-{}", "z".repeat(1_024));
        let body =
            serde_json::to_vec(&json!({secret.clone(): null, "data": []})).expect("test response must serialize");
        let error = parse_response_body(&body, "store-a", 10, Instant::now() + Duration::from_secs(60))
            .expect_err("invalid response must fail");
        let rendered = error.to_string();

        assert!(rendered.len() < 256);
        assert!(!rendered.contains(&secret));
        assert!(rendered.contains("line"));
        assert!(rendered.contains("column"));
    }
}
