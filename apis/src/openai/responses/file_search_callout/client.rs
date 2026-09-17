// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Callout client for vector store search.

use std::{
    fmt, io,
    net::SocketAddr,
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use bytes::Bytes;
use http::HeaderMap;
use praxis_core::connectivity::{PreparedSubrequest, PreparedTarget, is_private_ip, prepare_url_target};
use praxis_filter::{
    CalloutOutcome, CalloutResponse, FilterPipeline, FilteredSubrequestExecutor, RequestExtensions, StagedUpstream,
    StagedUpstreamFallback, SubRequest, SubrequestRuntime,
};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Value, value::RawValue};

use crate::{
    callout_policy::OnFailure,
    http_hop::{connection_nominates_header, is_hop_by_hop},
    openai::api_client::resource_url,
    subrequest::SubRequestClient,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default number of retained results per file-search call.
const DEFAULT_MAX_NUM_RESULTS: usize = 10;

/// Maximum number of concurrent search callouts.
pub(super) const MAX_CONCURRENT_SEARCHES: usize = 8;

/// OpenAI's maximum number of file-search results per call.
const MAX_NUM_RESULTS: usize = 50;

/// Maximum rendered query size per search request: 64 `KiB`.
pub(super) const MAX_QUERY_BYTES: usize = 65_536;

/// Maximum serialized search request body: 1 MiB.
pub(super) const MAX_SEARCH_REQUEST_BYTES: usize = 1_048_576;

/// Maximum vector store identifier size accepted for one URL segment.
pub(super) const MAX_VECTOR_STORE_ID_BYTES: usize = 512;

/// Allocation unit used by global collected-response admission.
const RESPONSE_BODY_BUDGET_UNIT_BYTES: usize = 1_048_576; // 1 MiB

/// Charge wire bodies, parser scratch, and retained decoded storage.
const RESPONSE_ADMISSION_WIRE_MULTIPLIER: usize = 4;

/// Decoded-storage headroom reserved for each response in one execution.
const RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES: usize = 65_536; // 64 KiB

/// Conservative fixed storage charge for one retained result.
const DECODED_RESULT_BYTES: usize = 256;

/// Conservative fixed storage charge for one retained content chunk.
const DECODED_CONTENT_CHUNK_BYTES: usize = 128;

/// Conservative fixed storage charge for one retained attribute value.
const DECODED_ATTRIBUTE_VALUE_BYTES: usize = 128;

/// Conservative fixed storage charge for one retained attribute object member.
const DECODED_ATTRIBUTE_MEMBER_BYTES: usize = 128;

/// Round owned strings and keys to conservative allocator-sized units.
const DECODED_STRING_ROUND_BYTES: usize = 16;

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

    /// Ranking options translated from the Responses shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ranking_options: Option<SearchRankingOptions<'a>>,

    /// The backend must not perform a second query rewrite.
    pub(crate) rewrite_query: bool,

    /// Enable keyword and vector retrieval for hybrid ranking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_mode: Option<&'static str>,
}

impl<'a> VectorStoreSearchRequest<'a> {
    /// Build the wire model from Responses ranking options.
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

/// Ranking options for the vector store search endpoint.
#[derive(Debug, Serialize)]
pub(crate) struct SearchRankingOptions<'a> {
    /// Explicit ranker, or weighted ranker for hybrid weights.
    #[serde(skip_serializing_if = "Option::is_none")]
    ranker: Option<&'a str>,

    /// Normalized semantic weight for the weighted ranker.
    #[serde(skip_serializing_if = "Option::is_none")]
    alpha: Option<f64>,

    /// Minimum score copied from the Responses request.
    #[serde(skip_serializing_if = "Option::is_none")]
    score_threshold: Option<&'a Value>,
}

/// Borrowed ranking translation plus the retrieval mode it requires.
struct TranslatedRankingOptions<'a> {
    /// Translated ranking options.
    options: Option<SearchRankingOptions<'a>>,

    /// Retrieval mode.
    search_mode: Option<&'static str>,
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
        D: Deserializer<'de>,
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

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        if v == "text" {
            Ok(ContentChunkType::Text)
        } else {
            Err(E::custom("unsupported vector-store content chunk type"))
        }
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
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

    /// The vector-store response exceeded the configured size limit.
    ///
    /// Surfaced by [`FilteredSubrequestExecutor::run_classified`] as
    /// [`CalloutOutcome::ResponseTooLarge`] and mapped by the loop owner to an
    /// HTTP 413 rejection, distinguishing an oversized upstream body from a
    /// generic callout failure (502).
    ResponseTooLarge {
        /// Observed response size that tripped the limit, when known.
        actual: Option<usize>,

        /// Effective per-response byte ceiling that was exceeded.
        limit: usize,

        /// Vector store ID.
        store_id: String,
    },
}

impl FileSearchError {
    /// Map this failure to the loop owner's terminal HTTP status and error code.
    ///
    /// An oversized response is a client-actionable `413 invalid_request_error`,
    /// mirroring the continuation-state ceiling; every other failure is an opaque
    /// upstream or callout error surfaced as `502 server_error`.
    pub(super) fn dispatch_status(&self) -> (u16, &'static str) {
        match self {
            Self::ResponseTooLarge { .. } => (413, "invalid_request_error"),
            Self::Callout { .. } | Self::Deserialize { .. } => (502, "server_error"),
        }
    }
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
            Self::ResponseTooLarge {
                actual,
                limit,
                store_id,
            } => match actual {
                Some(actual) => write!(
                    f,
                    "vector-store response from store {store_id:?} exceeded the {limit}-byte limit (observed {actual} bytes)"
                ),
                None => write!(
                    f,
                    "vector-store response from store {store_id:?} exceeded the {limit}-byte limit"
                ),
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
    /// Vector-store API base URL (trailing slash stripped).
    pub base_url: String,

    /// Shared sub-request transport driving the outbound chain.
    pub subrequest_client: SubRequestClient,

    /// Header names forwarded from the original request to the vector store.
    pub forward_header_names: Vec<http::HeaderName>,

    /// Whether one failed chunk stops scheduling later callouts.
    pub on_failure: OnFailure,

    /// Maximum response body size enforced by the sub-request executor.
    pub max_response_bytes: usize,

    /// Maximum successful response bytes retained across the fan-out.
    pub max_total_response_bytes: usize,

    /// Whole-call timeout, including response body collection.
    pub timeout: Duration,
}

/// Per-request binding of the outbound chain and downstream attributes.
///
/// A [`FileSearchCalloutFilter`](super::FileSearchCalloutFilter) builds one of
/// these from its bound outbound pipeline and the live request context, then
/// hands it to [`FileSearchClient::search`] so the whole fan-out routes through
/// the same filtered sub-request transport.
pub(crate) struct CalloutTransport<'a> {
    /// Prebuilt outbound filter chain every vector-store sub-request runs through.
    pub outbound: &'a Arc<FilterPipeline>,

    /// Downstream client attributes forwarded into each sub-request.
    pub downstream: SubrequestRuntime,
}

/// Client for vector store search API.
pub(crate) struct FileSearchClient {
    /// Vector-store API base URL (trailing slash stripped).
    base_url: String,

    /// Shared sub-request transport driving the outbound chain.
    subrequest_client: SubRequestClient,

    /// Header names forwarded from the original request to the vector store.
    forward_header_names: Vec<http::HeaderName>,

    /// Whether one failed chunk stops scheduling later callouts.
    on_failure: OnFailure,

    /// Maximum response body size enforced by the sub-request executor.
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
            base_url: config.base_url,
            subrequest_client: config.subrequest_client,
            forward_header_names: config.forward_header_names,
            on_failure: config.on_failure,
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
    pub async fn search(
        &self,
        specs: &[SearchSpec<'_>],
        call_count: usize,
        request_headers: &HeaderMap,
        transport: &CalloutTransport<'_>,
    ) -> SearchBatch {
        let mut batch = SearchBatch::new(call_count);
        let mut consumed_response_bytes = 0_usize;
        let mut deadline_recorded = false;
        let mut next_spec = 0_usize;
        let execution_started = Instant::now();
        // Read the SSRF policy once from the bound outbound pipeline. The
        // executor's transport pins the resolved literal address and so never
        // re-runs the private-IP check (literal targets bypass it), making the
        // `prepare_url_target` validation hook the sole runtime SSRF gate.
        let allow_private = transport.outbound.allow_private_upstreams();
        // One executor drives the whole fan-out: `run` takes `&self`, so the
        // concurrent sub-requests share a single transport and downstream
        // projection. Outbound headers are identical across specs, so they are
        // built once and cloned only at each owned `SubRequest` boundary.
        let executor = FilteredSubrequestExecutor::for_callout(
            self.subrequest_client.clone(),
            transport.downstream.clone(),
            0,
            self.max_response_bytes,
            self.timeout,
        );
        let outbound_headers = self.build_outbound_headers(request_headers);
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
                append_unprocessed_deadline_failures(&mut batch.failures, specs, next_spec);
                deadline_recorded = true;
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
            let futures = chunk.iter().map(|spec| {
                self.search_one(
                    spec,
                    execution_started,
                    Arc::clone(&admission),
                    &executor,
                    transport.outbound,
                    &outbound_headers,
                    allow_private,
                )
            });
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
                append_unprocessed_deadline_failures(&mut batch.failures, specs, next_spec);
                deadline_recorded = true;
                break;
            }
            if chunk_failed && self.on_failure == OnFailure::Closed {
                if let Some(remaining_specs) = specs.get(next_spec..) {
                    append_fail_closed_failures(&mut batch.failures, remaining_specs);
                }
                break;
            }
        }

        batch.sort_results();
        if !deadline_recorded && execution_started.elapsed() >= self.timeout {
            append_unprocessed_deadline_failures(&mut batch.failures, specs, next_spec);
        }
        batch
    }

    /// Search a single vector store.
    #[expect(
        clippy::too_many_arguments,
        reason = "one sub-request threads the shared executor, chain, and prepared headers"
    )]
    async fn search_one(
        &self,
        spec: &SearchSpec<'_>,
        execution_started: Instant,
        response_admission: Arc<ResponseAdmission>,
        executor: &FilteredSubrequestExecutor,
        outbound: &Arc<FilterPipeline>,
        outbound_headers: &HeaderMap,
        allow_private: bool,
    ) -> Result<SearchResponse, FileSearchError> {
        deadline_remaining(self.timeout, execution_started, spec.store_id)?;
        let request = self.build_request(spec, execution_started)?;
        deadline_remaining(self.timeout, execution_started, spec.store_id)?;
        let body = self
            .execute_request(
                request,
                spec.store_id,
                execution_started,
                executor,
                outbound,
                outbound_headers,
                allow_private,
            )
            .await?;
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

    /// Resolve and validate the vector-store URL under the shared deadline.
    ///
    /// Because the returned [`PreparedTarget`] is later staged as a
    /// [`StagedUpstream`] (plus a [`StagedUpstreamFallback`] carrying the full
    /// resolved set) pinning the resolved literal addresses, the executor's
    /// transport never re-runs DNS resolution, so this `prepare_url_target`
    /// validation hook is the sole runtime SSRF gate: unless the outbound
    /// pipeline opts into `insecure_options.allow_private_upstreams`, it rejects
    /// the whole resolved address set if any entry is private or reserved.
    /// Vetting the whole set together closes the resolve-then-dial race — every
    /// address the executor may dial (the primary, and any fallback the executor
    /// tries after a refused connection) was vetted here. The large resolver
    /// future is boxed so it stays off this call's stack frame.
    #[expect(
        clippy::too_many_arguments,
        reason = "target preparation threads the deadline, deadline clock, and SSRF policy"
    )]
    async fn prepare_search_target(
        &self,
        url: &str,
        store_id: &str,
        execution_started: Instant,
        deadline: Instant,
        allow_private: bool,
    ) -> Result<PreparedTarget, FileSearchError> {
        let remaining = deadline_remaining(self.timeout, execution_started, store_id)?;
        tokio::time::timeout(
            remaining,
            Box::pin(prepare_url_target(url, deadline, |addresses: &[SocketAddr]| {
                if !allow_private && addresses.iter().any(|address| is_private_ip(&address.ip())) {
                    return Err::<(), Box<dyn std::error::Error + Send + Sync>>(
                        "vector-store target resolved to a private or reserved address".into(),
                    );
                }
                Ok(())
            })),
        )
        .await
        .map_err(|_elapsed| execution_deadline_error(store_id))?
        .map_err(|_error| request_error(store_id, "vector-store target preparation failed"))
    }

    /// Execute a callout through the outbound chain with a deadline covering
    /// target preparation and body collection.
    ///
    /// The destination is prepared from the search URL and staged as a
    /// [`StagedUpstream`] (with a [`StagedUpstreamFallback`] carrying the full
    /// validated address set so a refused connection falls back to the next
    /// resolved address) so the outbound chain carries only cross-cutting
    /// filters and never resolves a cluster. The SSRF gate lives in
    /// [`Self::prepare_search_target`]. A streaming response is rejected —
    /// file-search requires a buffered body.
    #[expect(
        clippy::too_many_arguments,
        reason = "one sub-request threads the shared executor, chain, prepared headers, and SSRF policy"
    )]
    async fn execute_request(
        &self,
        request: PreparedSearchRequest,
        store_id: &str,
        execution_started: Instant,
        executor: &FilteredSubrequestExecutor,
        outbound: &Arc<FilterPipeline>,
        outbound_headers: &HeaderMap,
        allow_private: bool,
    ) -> Result<Bytes, FileSearchError> {
        let deadline = execution_started
            .checked_add(self.timeout)
            .ok_or_else(|| execution_deadline_error(store_id))?;
        let target = self
            .prepare_search_target(&request.url, store_id, execution_started, deadline, allow_private)
            .await?;
        let staged = StagedUpstream::from_prepared_target(&target)
            .map_err(|_error| request_error(store_id, "vector-store upstream preparation failed"))?;
        // Stage the full validated address set so the executor can fall back
        // past a refused connection to the next resolved address, preserving the
        // DNS behavior of the low-level transport. Built before `bind` consumes
        // the target.
        let fallback = StagedUpstreamFallback::from_prepared_target(&target);

        let prepared = target.bind(SubRequest {
            method: http::Method::POST,
            uri: http::Uri::default(),
            headers: outbound_headers.clone(),
            body: Bytes::from(request.body),
        });
        let mut extensions = RequestExtensions::default();
        extensions.insert(staged);
        extensions.insert(fallback);

        self.run_outbound(
            executor,
            outbound,
            &prepared,
            extensions,
            store_id,
            execution_started,
            deadline,
        )
        .await
    }

    /// Drive the bound sub-request through the outbound chain under the shared
    /// deadline and return its buffered body.
    ///
    /// Isolated from [`Self::execute_request`] so the large executor future,
    /// boxed onto the heap, stays within this call's stack frame instead of
    /// inflating the caller's.
    #[expect(
        clippy::too_many_arguments,
        reason = "one sub-request threads the shared executor, chain, bound request, and deadline clock"
    )]
    async fn run_outbound(
        &self,
        executor: &FilteredSubrequestExecutor,
        outbound: &Arc<FilterPipeline>,
        prepared: &PreparedSubrequest,
        extensions: RequestExtensions,
        store_id: &str,
        execution_started: Instant,
        deadline: Instant,
    ) -> Result<Bytes, FileSearchError> {
        let remaining = deadline_remaining(self.timeout, execution_started, store_id)?;
        let outcome = tokio::time::timeout(
            remaining,
            Box::pin(executor.run_classified(outbound, prepared.request(), extensions, deadline)),
        )
        .await
        .map_err(|_elapsed| execution_deadline_error(store_id))?
        .map_err(|_error| request_error(store_id, "vector-store filtered sub-request failed"))?;
        match outcome {
            CalloutOutcome::Response(response) => read_buffered_response(response, store_id),
            // `run_classified` preserves the typed size-overflow classification
            // (transport-level or executor-side) that `run` would collapse into an
            // opaque 502, so the loop owner can map it to an actionable HTTP 413.
            CalloutOutcome::ResponseTooLarge { actual, limit } => Err(FileSearchError::ResponseTooLarge {
                actual,
                limit,
                store_id: bounded_store_id(store_id),
            }),
            // `CalloutOutcome` is `#[non_exhaustive]`: fail closed on any future
            // classified outcome rather than silently treating it as a success.
            _ => Err(request_error(
                store_id,
                "vector-store returned an unrecognized callout outcome",
            )),
        }
    }

    /// Build the outbound header set: forwarded request headers plus a JSON
    /// content type, mirroring the previous shared API client.
    fn build_outbound_headers(&self, request_headers: &HeaderMap) -> HeaderMap {
        let mut map = HeaderMap::new();
        for name in &self.forward_header_names {
            if is_hop_by_hop(name.as_str()) || connection_nominates_header(request_headers, name) {
                continue;
            }
            if let Some(value) = request_headers.get(name) {
                map.insert(name.clone(), value.clone());
            }
        }
        map.remove(http::header::CONTENT_TYPE);
        map.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        map
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

        resource_url(&self.base_url, "v1/vector_stores", store_id, Some("search"))
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

    /// Fully encoded search URL.
    url: String,
}

/// Process-wide admission retained through aggregation and timed-out decoders.
#[derive(Debug)]
struct ResponseAdmission {
    /// Reserved aggregate-response byte units.
    _body_budget: tokio::sync::OwnedSemaphorePermit,
}

/// Remaining allocation budget for retained typed response values.
///
/// Retained results are first walked as borrowed `RawValue`s to validate their
/// schema and charge the estimated owned storage for strings, attributes, and
/// content, then deserialized into `SearchResult`. Those representations can
/// briefly coexist, so `RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES` also provides
/// headroom for the duplicate allocations and parser scratch during that handoff.
struct DecodedBudget {
    /// Bytes that may still be represented by owned decoded values.
    remaining: usize,
}

impl DecodedBudget {
    /// Create the per-response decoded allocation allowance.
    fn for_body(body_bytes: usize) -> Result<Self, &'static str> {
        let remaining = body_bytes
            .checked_add(RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES)
            .ok_or("decoded response budget is too large")?;
        Ok(Self { remaining })
    }

    /// Reserve a conservative amount before constructing an owned value.
    fn charge(&mut self, bytes: usize) -> Result<(), &'static str> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or("decoded response exceeds its allocation budget")?;
        Ok(())
    }

    /// Reserve an allocator-rounded owned string or object key.
    fn charge_string(&mut self, bytes: usize) -> Result<(), &'static str> {
        let rounded = bytes
            .checked_add(DECODED_STRING_ROUND_BYTES - 1)
            .ok_or("decoded response allocation accounting overflow")?
            / DECODED_STRING_ROUND_BYTES
            * DECODED_STRING_ROUND_BYTES;
        self.charge(rounded)
    }
}

/// Root response fields recognized without allocating map keys.
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ResponseField {
    /// Required result sequence.
    Data,

    /// Provider extension ignored for compatibility.
    #[serde(other)]
    Other,
}

/// Result fields recognized during allocation-free validation.
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ResultField {
    /// Optional arbitrary attributes.
    Attributes,

    /// Required content sequence.
    Content,

    /// Required file identifier.
    FileId,

    /// Required filename.
    Filename,

    /// Required relevance score.
    Score,

    /// Provider extension ignored for compatibility.
    #[serde(other)]
    Other,
}

/// Content fields recognized during allocation-free validation.
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ContentField {
    /// Required chunk type.
    #[serde(rename = "type")]
    ChunkType,

    /// Required chunk text.
    Text,

    /// Provider extension ignored for compatibility.
    #[serde(other)]
    Other,
}

/// Incremental root response decoder.
struct SearchResponseSeed<'a> {
    /// Shared budget charged by the retained result candidates.
    decoded_budget: &'a mut DecodedBudget,
}

impl<'de> DeserializeSeed<'de> for SearchResponseSeed<'_> {
    type Value = Vec<SearchResult>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(SearchResponseVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Root visitor that preserves required and duplicate `data` handling.
struct SearchResponseVisitor<'a> {
    /// Shared budget charged by the retained result candidates.
    decoded_budget: &'a mut DecodedBudget,
}

impl<'de> Visitor<'de> for SearchResponseVisitor<'_> {
    type Value = Vec<SearchResult>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store search response object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut data = None;
        while let Some(field) = map.next_key::<ResponseField>()? {
            match field {
                ResponseField::Data => {
                    if data.is_some() {
                        return Err(de::Error::duplicate_field("data"));
                    }
                    data = Some(map.next_value_seed(SearchResultsSeed {
                        decoded_budget: self.decoded_budget,
                    })?);
                },
                ResponseField::Other => {
                    map.next_value::<IgnoredAny>()?;
                },
            }
        }
        data.ok_or_else(|| de::Error::missing_field("data"))
    }
}

/// Incremental `data` sequence decoder.
struct SearchResultsSeed<'a> {
    /// Shared budget charged by the retained result candidates.
    decoded_budget: &'a mut DecodedBudget,
}

impl<'de> DeserializeSeed<'de> for SearchResultsSeed<'_> {
    type Value = Vec<SearchResult>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(SearchResultsVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Retain the first bounded candidates and allocation-free validate the rest.
struct SearchResultsVisitor<'a> {
    /// Shared budget charged by the retained result candidates.
    decoded_budget: &'a mut DecodedBudget,
}

impl<'de> Visitor<'de> for SearchResultsVisitor<'_> {
    type Value = Vec<SearchResult>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store search result array")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut retained = Vec::with_capacity(MAX_NUM_RESULTS);
        while let Some(raw) = seq.next_element::<&'de RawValue>()? {
            if retained.len() < MAX_NUM_RESULTS {
                validate_search_result(raw.get(), Some(self.decoded_budget))
                    .map_err(|_error| de::Error::custom("invalid retained vector-store search result"))?;
                let result = serde_json::from_str::<SearchResult>(raw.get()).map_err(|error| {
                    de::Error::custom(format!("invalid retained vector-store search result: {error}"))
                })?;
                retained.push(result);
            } else {
                validate_search_result(raw.get(), None)
                    .map_err(|_error| de::Error::custom("invalid discarded vector-store search result"))?;
            }
        }
        Ok(retained)
    }
}

/// Validate one result, optionally charging all retained owned storage.
struct ResultValidationSeed<'a> {
    /// Budget is absent for entries beyond the retained candidate set.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> DeserializeSeed<'de> for ResultValidationSeed<'_> {
    type Value = ();

    fn deserialize<D>(mut self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if let Some(budget) = self.decoded_budget.as_deref_mut() {
            budget.charge(DECODED_RESULT_BYTES).map_err(de::Error::custom)?;
        }
        deserializer.deserialize_map(ResultValidationVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Validate the typed result schema without constructing its owned values.
struct ResultValidationVisitor<'a> {
    /// Budget is absent for entries beyond the retained candidate set.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> Visitor<'de> for ResultValidationVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store search result object")
    }

    #[expect(
        clippy::too_many_lines,
        reason = "required and duplicate result fields share allocation accounting"
    )]
    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut attributes_seen = false;
        let mut content_seen = false;
        let mut file_id = None;
        let mut filename_seen = false;
        let mut score_seen = false;
        let mut attribute_file_id_bytes = None;

        while let Some(field) = map.next_key::<ResultField>()? {
            match field {
                ResultField::Attributes => {
                    if attributes_seen {
                        return Err(de::Error::duplicate_field("attributes"));
                    }
                    attributes_seen = true;
                    attribute_file_id_bytes = map.next_value_seed(OptionalAttributesSeed {
                        decoded_budget: self.decoded_budget.as_deref_mut(),
                    })?;
                },
                ResultField::Content => {
                    if content_seen {
                        return Err(de::Error::duplicate_field("content"));
                    }
                    content_seen = true;
                    map.next_value_seed(ContentSequenceSeed {
                        decoded_budget: self.decoded_budget.as_deref_mut(),
                    })?;
                },
                ResultField::FileId => {
                    if file_id.is_some() {
                        return Err(de::Error::duplicate_field("file_id"));
                    }
                    file_id = Some(map.next_value_seed(MeasuredStringSeed {
                        decoded_budget: self.decoded_budget.as_deref_mut(),
                    })?);
                },
                ResultField::Filename => {
                    if filename_seen {
                        return Err(de::Error::duplicate_field("filename"));
                    }
                    filename_seen = true;
                    map.next_value_seed(MeasuredStringSeed {
                        decoded_budget: self.decoded_budget.as_deref_mut(),
                    })?;
                },
                ResultField::Score => {
                    if score_seen {
                        return Err(de::Error::duplicate_field("score"));
                    }
                    score_seen = true;
                    map.next_value::<f64>()?;
                },
                ResultField::Other => {
                    map.next_value::<IgnoredAny>()?;
                },
            }
        }

        if !content_seen {
            return Err(de::Error::missing_field("content"));
        }
        let file_id = file_id.ok_or_else(|| de::Error::missing_field("file_id"))?;
        if !filename_seen {
            return Err(de::Error::missing_field("filename"));
        }
        if !score_seen {
            return Err(de::Error::missing_field("score"));
        }
        if !file_id.canonical_file_id
            && let (Some(bytes), Some(budget)) = (attribute_file_id_bytes, self.decoded_budget.as_deref_mut())
        {
            budget.charge_string(bytes).map_err(de::Error::custom)?;
        }

        Ok(())
    }
}

/// Result of inspecting a borrowed string.
#[derive(Clone, Copy)]
struct MeasuredString {
    /// Whether the string is a non-empty OpenAI file identifier.
    canonical_file_id: bool,
}

/// Validate a string while charging the allocation made by typed decoding.
struct MeasuredStringSeed<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> DeserializeSeed<'de> for MeasuredStringSeed<'_> {
    type Value = MeasuredString;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(MeasuredStringVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Borrow string contents even when Serde JSON must use transient escape scratch.
struct MeasuredStringVisitor<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl Visitor<'_> for MeasuredStringVisitor<'_> {
    type Value = MeasuredString;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string")
    }

    fn visit_str<E>(mut self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Some(budget) = self.decoded_budget.as_deref_mut() {
            budget.charge_string(v.len()).map_err(E::custom)?;
        }
        Ok(MeasuredString {
            canonical_file_id: v.strip_prefix("file-").is_some_and(|suffix| !suffix.is_empty()),
        })
    }

    fn visit_borrowed_str<E>(self, v: &'_ str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(v)
    }
}

/// Treat JSON `null` attributes as `None`, matching `Option<Value>`.
struct OptionalAttributesSeed<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> DeserializeSeed<'de> for OptionalAttributesSeed<'_> {
    type Value = Option<usize>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_option(OptionalAttributesVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Optional attribute wrapper used to avoid charging a non-retained null value.
struct OptionalAttributesVisitor<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> Visitor<'de> for OptionalAttributesVisitor<'_> {
    type Value = Option<usize>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an arbitrary JSON attribute value or null")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        AttributeValueSeed {
            decoded_budget: self.decoded_budget,
            capture_canonical_file_id: false,
            direct_attributes_object: true,
        }
        .deserialize(deserializer)
    }
}

/// Recursively validate arbitrary attributes and charge their owned `Value` shape.
struct AttributeValueSeed<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,

    /// Return a canonical string length for direct `attributes.file_id`.
    capture_canonical_file_id: bool,

    /// Recognize `file_id` only on the root attributes object.
    direct_attributes_object: bool,
}

impl<'de> DeserializeSeed<'de> for AttributeValueSeed<'_> {
    type Value = Option<usize>;

    fn deserialize<D>(mut self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if let Some(budget) = self.decoded_budget.as_deref_mut() {
            budget
                .charge(DECODED_ATTRIBUTE_VALUE_BYTES)
                .map_err(de::Error::custom)?;
        }
        deserializer.deserialize_any(AttributeValueVisitor {
            decoded_budget: self.decoded_budget,
            capture_canonical_file_id: self.capture_canonical_file_id,
            direct_attributes_object: self.direct_attributes_object,
        })
    }
}

/// Allocation-free arbitrary-JSON visitor used by the attribute preflight.
struct AttributeValueVisitor<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,

    /// Return a canonical string length for direct `attributes.file_id`.
    capture_canonical_file_id: bool,

    /// Recognize `file_id` only on the root attributes object.
    direct_attributes_object: bool,
}

/// Implement an allocation-free scalar attribute visitor method.
macro_rules! visit_attribute_scalar {
    ($method:ident, $type:ty) => {
        fn $method<E>(self, _value: $type) -> Result<Self::Value, E> {
            Ok(None)
        }
    };
}

impl<'de> Visitor<'de> for AttributeValueVisitor<'_> {
    type Value = Option<usize>;

    visit_attribute_scalar!(visit_bool, bool);

    visit_attribute_scalar!(visit_i64, i64);

    visit_attribute_scalar!(visit_u64, u64);

    visit_attribute_scalar!(visit_f64, f64);

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an arbitrary JSON attribute value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_str<E>(mut self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Some(budget) = self.decoded_budget.as_deref_mut() {
            budget.charge_string(v.len()).map_err(E::custom)?;
        }
        Ok(
            (self.capture_canonical_file_id && v.strip_prefix("file-").is_some_and(|suffix| !suffix.is_empty()))
                .then_some(v.len()),
        )
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(v)
    }

    fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq
            .next_element_seed(AttributeValueSeed {
                decoded_budget: self.decoded_budget.as_deref_mut(),
                capture_canonical_file_id: false,
                direct_attributes_object: false,
            })?
            .is_some()
        {}
        Ok(None)
    }

    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut captured_file_id = None;
        while let Some(is_file_id) = map.next_key_seed(AttributeKeySeed {
            decoded_budget: self.decoded_budget.as_deref_mut(),
        })? {
            if let Some(budget) = self.decoded_budget.as_deref_mut() {
                budget
                    .charge(DECODED_ATTRIBUTE_MEMBER_BYTES)
                    .map_err(de::Error::custom)?;
            }
            let candidate = map.next_value_seed(AttributeValueSeed {
                decoded_budget: self.decoded_budget.as_deref_mut(),
                capture_canonical_file_id: self.direct_attributes_object && is_file_id,
                direct_attributes_object: false,
            })?;
            if self.direct_attributes_object && is_file_id {
                captured_file_id = candidate;
            }
        }
        Ok(captured_file_id)
    }
}

/// Borrow and charge an attribute object key without retaining it.
struct AttributeKeySeed<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> DeserializeSeed<'de> for AttributeKeySeed<'_> {
    type Value = bool;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(AttributeKeyVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Attribute key visitor returning only whether the key is `file_id`.
struct AttributeKeyVisitor<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl Visitor<'_> for AttributeKeyVisitor<'_> {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an attribute object key")
    }

    fn visit_str<E>(mut self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if let Some(budget) = self.decoded_budget.as_deref_mut() {
            budget.charge_string(v.len()).map_err(E::custom)?;
        }
        Ok(v == "file_id")
    }

    fn visit_borrowed_str<E>(self, v: &'_ str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(v)
    }
}

/// Validate and account for a result content sequence.
struct ContentSequenceSeed<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> DeserializeSeed<'de> for ContentSequenceSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ContentSequenceVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Consume every content chunk without constructing a temporary vector.
struct ContentSequenceVisitor<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> Visitor<'de> for ContentSequenceVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store content array")
    }

    fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq
            .next_element_seed(ContentChunkSeed {
                decoded_budget: self.decoded_budget.as_deref_mut(),
            })?
            .is_some()
        {}
        Ok(())
    }
}

/// Validate and account for one content chunk.
struct ContentChunkSeed<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> DeserializeSeed<'de> for ContentChunkSeed<'_> {
    type Value = ();

    fn deserialize<D>(mut self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if let Some(budget) = self.decoded_budget.as_deref_mut() {
            budget.charge(DECODED_CONTENT_CHUNK_BYTES).map_err(de::Error::custom)?;
        }
        deserializer.deserialize_map(ContentChunkVisitor {
            decoded_budget: self.decoded_budget,
        })
    }
}

/// Validate required and duplicate content fields without retaining them.
struct ContentChunkVisitor<'a> {
    /// Budget is absent while validating discarded results.
    decoded_budget: Option<&'a mut DecodedBudget>,
}

impl<'de> Visitor<'de> for ContentChunkVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a vector-store content chunk object")
    }

    #[expect(
        clippy::too_many_lines,
        reason = "required and duplicate chunk fields are validated together"
    )]
    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut chunk_type_seen = false;
        let mut text_seen = false;
        while let Some(field) = map.next_key::<ContentField>()? {
            match field {
                ContentField::ChunkType => {
                    if chunk_type_seen {
                        return Err(de::Error::duplicate_field("type"));
                    }
                    chunk_type_seen = true;
                    map.next_value::<ContentChunkType>()?;
                },
                ContentField::Text => {
                    if text_seen {
                        return Err(de::Error::duplicate_field("text"));
                    }
                    text_seen = true;
                    map.next_value_seed(MeasuredStringSeed {
                        decoded_budget: self.decoded_budget.as_deref_mut(),
                    })?;
                },
                ContentField::Other => {
                    map.next_value::<IgnoredAny>()?;
                },
            }
        }
        if !chunk_type_seen {
            return Err(de::Error::missing_field("type"));
        }
        if !text_seen {
            return Err(de::Error::missing_field("text"));
        }
        Ok(())
    }
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

/// Translate the OpenAI Responses hybrid-search shape to the vector store typed schema.
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
            options: score_threshold.is_some().then_some(SearchRankingOptions {
                ranker: None,
                alpha: None,
                score_threshold,
            }),
            search_mode: None,
        }),
        Some(hybrid) => translate_hybrid_ranking_options(hybrid, score_threshold),
    }
}

/// Map OpenAI's default selectors to the configured default ranker.
fn validate_ranker(source: &serde_json::Map<String, Value>) -> Result<(), &'static str> {
    match source.get("ranker") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(ranker)) if matches!(ranker.as_str(), "auto" | "default-2024-11-15") => Ok(()),
        Some(Value::String(_ranker)) => Err("ranking_options.ranker is not supported"),
        Some(_) => Err("ranking_options.ranker must be a string"),
    }
}

/// Translate Responses hybrid weights to the weighted ranker shape.
fn translate_hybrid_ranking_options<'a>(
    hybrid: &'a Value,
    score_threshold: Option<&'a Value>,
) -> Result<TranslatedRankingOptions<'a>, &'static str> {
    let hybrid = hybrid
        .as_object()
        .ok_or("ranking_options.hybrid_search must be an object")?;
    let alpha = normalized_hybrid_alpha(hybrid)?;

    Ok(TranslatedRankingOptions {
        options: Some(SearchRankingOptions {
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

/// Normalize required Responses hybrid weights into the semantic alpha.
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
    let aggregate_wire_bytes = max_response_bytes
        .checked_mul(spec_count)
        .ok_or("response admission is too large")?
        .min(max_total_response_bytes);
    let decoded_overhead_bytes = RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES
        .checked_mul(spec_count)
        .ok_or("response admission is too large")?;
    let admitted_bytes = aggregate_wire_bytes
        .checked_mul(RESPONSE_ADMISSION_WIRE_MULTIPLIER)
        .and_then(|bytes| bytes.checked_add(decoded_overhead_bytes))
        .ok_or("response admission is too large")?;
    let units = u32::try_from(admitted_bytes.div_ceil(RESPONSE_BODY_BUDGET_UNIT_BYTES))
        .map_err(|_overflow| "response admission is too large")?;
    if usize::try_from(units).map_or(true, |units| units > GLOBAL_RESPONSE_BODY_BUDGET_UNITS) {
        return Err("response admission exceeds the process-wide budget");
    }
    Ok(units)
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

/// Return the buffered success body, or a typed error for a non-2xx status or a
/// streaming response — file-search requires a buffered body.
fn read_buffered_response(response: CalloutResponse, store_id: &str) -> Result<Bytes, FileSearchError> {
    match response {
        CalloutResponse::Buffered(response) => {
            if !(200..300).contains(&response.status) {
                return Err(request_error(
                    store_id,
                    format!("vector-store returned status {}", response.status),
                ));
            }
            Ok(response.body)
        },
        CalloutResponse::Streaming { .. } => Err(request_error(
            store_id,
            "vector-store returned a streaming response; file-search requires a buffered body",
        )),
    }
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
    body: Bytes,
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
    let mut task = tokio::task::spawn_blocking(move || {
        let _response_admission = response_admission;
        let _decode_slot = decode_slot;
        parse_response_body(&body, &parse_store_id, result_limit)
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

/// Parse one API response without retaining its response buffer.
fn parse_response_body(body: &[u8], store_id: &str, result_limit: usize) -> Result<SearchResponse, FileSearchError> {
    let body_bytes = body.len();
    let data = deserialize_search_results(body, result_limit).map_err(|error| FileSearchError::Deserialize {
        body_bytes,
        line: error.line(),
        column: error.column(),
        store_id: store_id.to_owned(),
    })?;
    Ok(SearchResponse { body_bytes, data })
}

/// Deserialize the response page, apply file-ID fixups, and retain top-k.
fn deserialize_search_results(body: &[u8], result_limit: usize) -> Result<Vec<SearchResult>, serde_json::Error> {
    let mut decoded_budget =
        DecodedBudget::for_body(body.len()).map_err(|message| serde_json::Error::io(io::Error::other(message)))?;
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let mut data = SearchResponseSeed {
        decoded_budget: &mut decoded_budget,
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    for result in &mut data {
        fixup_file_id(result);
    }
    data.sort_by(|left, right| right.score.total_cmp(&left.score));
    data.truncate(result_limit);
    Ok(data)
}

/// Validate one borrowed result without allocating its typed representation.
fn validate_search_result(raw: &str, decoded_budget: Option<&mut DecodedBudget>) -> Result<(), serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    ResultValidationSeed { decoded_budget }.deserialize(&mut deserializer)?;
    deserializer.end()
}

/// Replace the backend's internal document UUID with the OpenAI Files API ID.
///
/// Some backends return their internal document UUID as the top-level `file_id` and
/// retain the source `file-*` identifier in attributes. Canonical responses
/// already carrying a `file-*` ID remain authoritative.
fn fixup_file_id(result: &mut SearchResult) {
    if result
        .file_id
        .strip_prefix("file-")
        .is_some_and(|suffix| !suffix.is_empty())
    {
        return;
    }
    if let Some(canonical) = result
        .attributes
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|attributes| attributes.get("file_id"))
        .and_then(Value::as_str)
        .filter(|candidate| candidate.strip_prefix("file-").is_some_and(|suffix| !suffix.is_empty()))
    {
        result.file_id = canonical.to_owned();
    }
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
        // An oversized response was rejected before any body was retained, so it
        // charges nothing against the aggregate budget, like a callout failure.
        Err(FileSearchError::Callout { .. } | FileSearchError::ResponseTooLarge { .. }) => 0,
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

/// Mark only specs at or after the scheduler's first unprocessed position.
fn append_unprocessed_deadline_failures(failures: &mut Vec<SearchFailure>, specs: &[SearchSpec<'_>], next_spec: usize) {
    if let Some(unprocessed) = specs.get(next_spec..) {
        append_deadline_failures(failures, unprocessed);
    }
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
        time::Duration,
    };

    use serde_json::{Value, json};

    use super::{
        FileSearchError, GLOBAL_RESPONSE_BODY_BUDGET_UNITS, RESPONSE_BODY_BUDGET_UNIT_BYTES,
        RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES, SearchResult, SearchSpec, VectorStoreSearchRequest,
        append_unprocessed_deadline_failures, deserialize_search_results, merge_top_results, parse_response_body,
        response_admission_units,
    };

    const MINIMAL_RESULT: &[u8] = br#"{"content":[],"file_id":"","filename":"","score":0}"#;

    fn decode(body: &[u8], limit: usize) -> Result<Vec<SearchResult>, serde_json::Error> {
        deserialize_search_results(body, limit)
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

    fn search_spec(call_index: usize, store_id: &str) -> SearchSpec<'_> {
        SearchSpec {
            call_index,
            filters: None,
            max_num_results: None,
            query: "query",
            ranking_options: None,
            store_id,
        }
    }

    #[test]
    fn deadline_failures_only_cover_unprocessed_specs() {
        let specs = [
            search_spec(0, "vs-completed-a"),
            search_spec(1, "vs-completed-b"),
            search_spec(2, "vs-unprocessed"),
        ];
        let mut failures = Vec::new();

        append_unprocessed_deadline_failures(&mut failures, &specs, 2);

        assert_eq!(failures.len(), 1);
        let failure = failures.first().expect("one unprocessed spec must fail");
        assert_eq!(failure.call_index, 2);
        assert!(matches!(
            &failure.error,
            FileSearchError::Callout { store_id, .. } if store_id == "vs-unprocessed"
        ));

        failures.clear();
        append_unprocessed_deadline_failures(&mut failures, &specs, specs.len());
        assert!(
            failures.is_empty(),
            "a completed final chunk must not be failed retroactively"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn aggregate_admission_serializes_whole_many_spec_executions() {
        let units = response_admission_units(10_485_760, 67_108_864, 64).expect("admission must fit");
        assert_eq!(units, 260);
        let budget = Arc::new(tokio::sync::Semaphore::new(GLOBAL_RESPONSE_BODY_BUDGET_UNITS));
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
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn admission_charge_covers_wire_and_per_response_decode_allowances() {
        let per_call = 10_485_760;
        let total = 67_108_864;
        let spec_count = 64;
        let units = usize::try_from(response_admission_units(per_call, total, spec_count).expect("admission must fit"))
            .expect("admission units must fit usize");
        let charged = units * RESPONSE_BODY_BUDGET_UNIT_BYTES;
        let required = 4 * total + RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES * spec_count;

        assert!(charged >= required);
        assert!(charged - required < RESPONSE_BODY_BUDGET_UNIT_BYTES);
        assert!(units <= GLOBAL_RESPONSE_BODY_BUDGET_UNITS);
        assert!(response_admission_units(usize::MAX, usize::MAX, 2).is_err());
    }

    #[test]
    fn deserializer_retains_top_ranked_chunks() {
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
                .any(|result| result.file_id == "file-a" && result.score == 0.9),
            "highest-scored file-a chunk must be retained"
        );
        assert!(
            decoded.iter().any(|result| result.file_id == "file-b"),
            "file-b must be retained"
        );
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
    fn openai_default_rankers_use_configured_default() {
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
    fn deserializer_requires_data_array() {
        assert!(decode(b"{}", 10).is_err(), "empty object must fail without data field");
        assert!(decode(br#"{"data":null}"#, 10).is_err(), "null data must fail");
        assert!(
            decode(br#"{"data":[],"data":[]}"#, 10).is_err(),
            "duplicate data must fail"
        );
        assert!(
            decode(br#"{"object":"list","data":[],"has_more":false}"#, 10)
                .expect("unknown response fields remain compatible")
                .is_empty()
        );
    }

    #[test]
    fn retained_and_discarded_results_remain_schema_validated() {
        assert!(
            decode(br#"{"data":[{"file_id":"file-a","filename":"a.txt","score":0.5}]}"#, 10).is_err(),
            "a retained result missing content must fail"
        );

        let mut body = br#"{"data":["#.to_vec();
        for index in 0..50 {
            if index != 0 {
                body.push(b',');
            }
            body.extend_from_slice(MINIMAL_RESULT);
        }
        body.extend_from_slice(br#",{"content":[],"file_id":"","filename":"","score":"bad"}]}"#);

        assert!(
            decode(&body, 10).is_err(),
            "a malformed result after entry 50 must fail"
        );
    }

    #[test]
    fn entries_after_fifty_do_not_enter_the_ranked_candidate_set() {
        let mut results = (0..50)
            .map(|index| {
                json!({
                    "content": [],
                    "file_id": format!("file-{index}"),
                    "filename": format!("{index}.txt"),
                    "score": f64::from(index),
                })
            })
            .collect::<Vec<_>>();
        results.push(json!({
            "attributes": {"file_id": "file-discarded"},
            "content": [],
            "file_id": "internal-discarded",
            "filename": "discarded.txt",
            "score": 10_000.0,
        }));
        let body = serde_json::to_vec(&json!({"data": results})).expect("test response must serialize");

        let decoded = decode(&body, 5).expect("all results are schema-valid");

        assert_eq!(decoded.len(), 5);
        assert_eq!(decoded.first().map(|result| result.file_id.as_str()), Some("file-49"));
        assert!(decoded.iter().all(|result| result.file_id != "file-discarded"));
    }

    #[test]
    fn deserializer_accepts_primitive_attribute_values() {
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
    fn deserializer_uses_source_file_id_for_internal_documents() {
        let body = serde_json::to_vec(&json!({"data":[
            {"attributes":{"file_id":"file-source"},"content":[{"type":"text","text":"fallback"}],
             "file_id":"83441278-02e0-44bb-b385-892a1d4680c5","filename":"report.txt","score":0.9},
            {"attributes":{"file_id":"file-shadow"},"content":[{"type":"text","text":"canonical"}],
             "file_id":"file-canonical","filename":"report.txt","score":0.8}
        ]}))
        .expect("test response must serialize");
        let decoded = decode(&body, 10).expect("search result IDs must decode");

        let mut decoded = decoded.into_iter();
        assert_eq!(decoded.next().map(|r| r.file_id).as_deref(), Some("file-source"));
        assert_eq!(decoded.next().map(|r| r.file_id).as_deref(), Some("file-canonical"));
    }

    #[test]
    fn escaped_retained_strings_decode_within_the_allocation_budget() {
        let escaped = r"\u0061".repeat(20_000);
        let body = format!(
            r#"{{"data":[{{"content":[{{"type":"text","text":"{escaped}"}}],"file_id":"file-a","filename":"a.txt","score":1}}]}}"#
        );

        let decoded = decode(body.as_bytes(), 1).expect("escaped string must fit its charged budget");

        let text = decoded
            .first()
            .and_then(|result| result.content.first())
            .map(|chunk| chunk.text.as_str());
        assert_eq!(text.map(str::len), Some(20_000));
    }

    #[test]
    fn oversized_single_result_attributes_fail_preflight() {
        let mut nested = Value::Null;
        for _ in 0..64 {
            let mut values = vec![Value::Null; 8];
            values.push(nested);
            nested = Value::Array(values);
        }
        let body = serde_json::to_vec(&json!({
            "data": [{
                "attributes": nested,
                "content": [],
                "file_id": "file-a",
                "filename": "a.txt",
                "score": 1.0,
            }]
        }))
        .expect("test response must serialize");

        assert!(
            decode(&body, 1).is_err(),
            "decoded attribute structure must not exceed body bytes plus fixed headroom"
        );
    }

    #[test]
    fn oversized_single_result_content_cardinality_fails_preflight() {
        let content = (0..1_000)
            .map(|_| json!({"type": "text", "text": ""}))
            .collect::<Vec<_>>();
        let body = serde_json::to_vec(&json!({
            "data": [{
                "content": content,
                "file_id": "file-a",
                "filename": "a.txt",
                "score": 1.0,
            }]
        }))
        .expect("test response must serialize");

        assert!(
            decode(&body, 1).is_err(),
            "retained chunk storage must not exceed body bytes plus fixed headroom"
        );
    }

    #[test]
    fn high_cardinality_decode_has_constant_scale_peak_allocation() {
        let result_count = 100_000;
        let mut body = Vec::with_capacity(MINIMAL_RESULT.len() * result_count + result_count + 10);
        body.extend_from_slice(br#"{"data":["#);
        for index in 0..result_count {
            if index != 0 {
                body.push(b',');
            }
            body.extend_from_slice(MINIMAL_RESULT);
        }
        body.extend_from_slice(b"]}");

        let mut decoded_len = 0;
        let allocations = allocation_counter::measure(|| {
            let decoded = decode(&body, 50).expect("minimal results must decode");
            decoded_len = decoded.len();
            std::hint::black_box(&decoded);
        });

        assert_eq!(decoded_len, 50);
        assert!(
            allocations.bytes_max <= RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES as u64,
            "peak live decoder allocation must remain constant-scale: {allocations:?}"
        );
        assert!(
            allocations.bytes_max <= (body.len() + RESPONSE_RETAINED_DECODE_OVERHEAD_BYTES) as u64,
            "peak live decoder allocation must fit the charged decoded allowance: {allocations:?}"
        );
    }

    #[test]
    fn decode_error_retains_only_location_metadata() {
        let body = br#"{"data":[{"content":[{"type":"text","text":"x"}],"file_id":"file-a","filename":"a.txt","score":"NaN"}]}"#;
        let error = parse_response_body(body, "store-a", 10).expect_err("invalid response must fail");
        let rendered = error.to_string();

        assert!(rendered.len() < 256, "error must be compact");
        assert!(rendered.contains("line"), "error must include line");
        assert!(rendered.contains("column"), "error must include column");
    }
}
