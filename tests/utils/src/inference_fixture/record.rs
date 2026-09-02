// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Live provider recording for two-sided inference fixtures.

use std::{
    collections::BTreeMap,
    fmt, io,
    net::{Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::{Bytes, BytesMut};
use futures::{StreamExt as _, stream};
use http::{HeaderMap, HeaderName, HeaderValue, Request, Response, StatusCode, header};
use http_body_util::{BodyExt as _, Full, Limited, StreamBody};
use hyper::body::{Body as _, Frame, Incoming};
use serde::Serialize;
use tokio::{sync::Notify, time::Instant};
use url::Host;

use super::{
    FixtureError, FixtureProvenance, InferenceScenario, MAX_INFERENCE_TURNS, NormalizationMetadata, ProvenanceKind,
    RecordedBody, RecordedExchange, RecordedRequest, RecordedResponse, RedactionRules, WIRE_FIXTURE_VERSION,
    WireFixture, WireTurn,
    bounds::{
        MAX_SCENARIO_REQUEST_BODY_BYTES, MAX_SCRIPTED_RESPONSE_BODY_BYTES, parse_request_body, parse_response_body,
        validate_request_body, validate_response_body,
    },
    header_policy::{
        contains_configured_credential, headers_contain_configured_credential, http_fixture_headers,
        provider_request_headers, provider_response_headers, validate_provider_headers,
    },
    http_server::{RecordingBody, RecordingHandler, RecordingHttpServer, ResponseDelivery, observe_response_delivery},
    replay::{
        build_replay_config, load_and_validate_config_source, send_recorded_request_with_header, validate_expectation,
        validate_scenario_requests,
    },
    sanitize::{sanitize_fixture_preserving_structure, validate_commit_safe_with_rules_and_credentials},
    schema::{
        DocumentResourceUsage, DocumentValidationLimits, LIVE_CAPTURE_STRUCTURE_LIMITS, MAX_FIXTURE_DOCUMENT_BYTES,
        measure_json_value_with_limits,
    },
};
use crate::{free_port_guard, start_proxy_no_wait, wait_for_tcp};

/// Maximum time allowed for one provider attempt to reach a terminal state.
const RECORDING_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Number of body chunks permitted between the provider reader and Hyper.
const RESPONSE_CHANNEL_CAPACITY: usize = 4;

/// Private hop header required by the credential-bearing recording listener.
static RECORDER_CAPABILITY_HEADER: HeaderName = HeaderName::from_static("x-inference-recorder-capability");

/// Maximum time allowed to receive one complete provider request body.
const RECORDING_REQUEST_BODY_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum wall-clock time allowed for one outbound provider request and body.
const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Maximum compact serialized bytes retained by either live capture leg.
///
/// Two legs consume at most two thirds of the loader ceiling, leaving the
/// final third for fixture metadata and pretty-printing whitespace.
const MAX_LIVE_CAPTURE_SERIALIZED_BYTES: usize = MAX_FIXTURE_DOCUMENT_BYTES / 3;

/// Shared compact and structural ceiling for one capture leg.
const LIVE_CAPTURE_BUDGET: CaptureBudget = CaptureBudget {
    serialized_bytes: MAX_LIVE_CAPTURE_SERIALIZED_BYTES,
    structure: LIVE_CAPTURE_STRUCTURE_LIMITS,
};

/// Compact JSON bytes around the request and response values of one exchange.
const RECORDED_EXCHANGE_ENVELOPE_BYTES: usize = b"{\"request\":".len() + b",\"response\":".len() + b"}".len();

/// A validated live provider destination and its in-memory outbound headers.
pub struct ProviderTarget {
    /// Provider provenance written to the sanitized fixture.
    pub provider: String,
    /// Model bound into exact `${MODEL}` scenario values.
    pub model: String,
    /// HTTP or HTTPS provider base URL, optionally including a path prefix.
    pub base_url: reqwest::Url,
    /// Headers injected only into outbound provider requests. Mark values under
    /// provider-specific credential names with [`HeaderValue::set_sensitive`].
    pub outbound_headers: HeaderMap,
}

impl fmt::Debug for ProviderTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderTarget")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("base_url", &"[redacted provider URL]")
            .field("outbound_headers", &"[redacted]")
            .finish()
    }
}

/// Starts loopback recording proxies for live provider exchanges.
pub struct RecordingProxy;

impl RecordingProxy {
    /// Starts one recording proxy after validating the complete provider target.
    ///
    /// # Errors
    ///
    /// Returns an opaque error if the target is invalid, the Rustls Reqwest
    /// client cannot be built, or the loopback listener cannot start.
    pub async fn start(target: ProviderTarget) -> Result<RecordingProxyGuard, FixtureError> {
        Self::start_bounded(target, MAX_INFERENCE_TURNS).await
    }

    /// Starts one recorder with an exact upper bound on provider attempts.
    async fn start_bounded(target: ProviderTarget, max_attempts: usize) -> Result<RecordingProxyGuard, FixtureError> {
        tokio::task::spawn_blocking(move || Self::start_inner(target, max_attempts, None, None))
            .await
            .map_err(|_panic| runtime_error("recording proxy startup task failed"))?
    }

    /// Starts a recorder with one additional test-only Rustls root certificate.
    #[cfg(test)]
    async fn start_with_test_root(
        target: ProviderTarget,
        root: reqwest::Certificate,
    ) -> Result<RecordingProxyGuard, FixtureError> {
        tokio::task::spawn_blocking(move || Self::start_inner(target, MAX_INFERENCE_TURNS, Some(root), None))
            .await
            .map_err(|_panic| runtime_error("recording proxy startup task failed"))?
    }

    /// Starts a recorder with a deterministic test-only connection completion notification.
    #[cfg(test)]
    async fn start_with_connection_finished(
        target: ProviderTarget,
        connection_finished: Arc<Notify>,
    ) -> Result<RecordingProxyGuard, FixtureError> {
        tokio::task::spawn_blocking(move || {
            Self::start_inner(target, MAX_INFERENCE_TURNS, None, Some(connection_finished))
        })
        .await
        .map_err(|_panic| runtime_error("recording proxy startup task failed"))?
    }

    /// Shared startup implementation with optional local test trust.
    fn start_inner(
        target: ProviderTarget,
        max_attempts: usize,
        root: Option<reqwest::Certificate>,
        connection_finished: Option<Arc<Notify>>,
    ) -> Result<RecordingProxyGuard, FixtureError> {
        validate_target(&target)?;
        if max_attempts == 0 || max_attempts > MAX_INFERENCE_TURNS {
            return Err(runtime_error("recording proxy attempt limit is invalid"));
        }
        let client = build_provider_client(root)?;
        let hop_client = build_recorder_hop_client()?;
        let capability = generate_recorder_capability()?;
        let shared = Arc::new(RecorderShared {
            client,
            hop_client,
            state: Mutex::new(RecorderState {
                attempts: Vec::new(),
                excess_attempt: false,
                retained: CaptureUsage::default(),
            }),
            target,
            capability,
            max_attempts,
            retained_limit: LIVE_CAPTURE_BUDGET,
            terminal: Notify::new(),
        });
        let handler_shared = Arc::clone(&shared);
        let handler: RecordingHandler = Arc::new(move |request| {
            let shared = Arc::clone(&handler_shared);
            Box::pin(async move { handle_provider_request(request, shared).await })
        });
        let server = RecordingHttpServer::start(handler, connection_finished)?;
        Ok(RecordingProxyGuard { server, shared })
    }
}

/// RAII guard for one live recording proxy and its completed captures.
///
/// Successful [`RecordingProxyGuard::finish`] tears down the runtime, joins its
/// OS thread, and releases the listener. Drop is bounded best-effort cleanup;
/// an unrecoverable timeout is logged by the internal server owner.
pub struct RecordingProxyGuard {
    /// Joined HTTP listener runtime.
    server: RecordingHttpServer,
    /// Provider target, client, capture state, and terminal notification.
    shared: Arc<RecorderShared>,
}

impl fmt::Debug for RecordingProxyGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordingProxyGuard")
            .field("addr", &self.addr())
            .finish_non_exhaustive()
    }
}

impl RecordingProxyGuard {
    /// Returns the recorder's loopback socket address.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.server.addr()
    }

    /// Sends one already-built request to this exact recorder socket.
    ///
    /// The per-run capability is marked sensitive and must not be persisted in
    /// fixtures or forwarded beyond the recording proxy.
    ///
    /// # Errors
    ///
    /// Returns an opaque error without attaching the capability when the
    /// request destination is not this recorder's exact loopback address.
    pub async fn send(&self, mut request: reqwest::Request) -> Result<reqwest::Response, FixtureError> {
        let destination = request.url();
        let target_ip = destination.host_str().and_then(|host| host.parse().ok());
        if destination.scheme() != "http"
            || target_ip != Some(self.addr().ip())
            || destination.port_or_known_default() != Some(self.addr().port())
        {
            return Err(runtime_error("recording request destination is invalid"));
        }
        request.headers_mut().remove(&RECORDER_CAPABILITY_HEADER);
        request
            .headers_mut()
            .insert(RECORDER_CAPABILITY_HEADER.clone(), self.shared.capability.clone());
        self.shared
            .hop_client
            .execute(request)
            .await
            .map_err(|_source| FixtureError::ReplayHttp)
    }

    /// Returns the per-run capability sent only across the trusted Praxis hop.
    fn capability(&self) -> &HeaderValue {
        &self.shared.capability
    }

    /// Stops and joins the recorder, then returns exact completed exchanges.
    ///
    /// # Errors
    ///
    /// Returns an opaque error for incomplete lifecycle accounting, a provider
    /// failure before response start, or an incomplete response capture.
    pub fn finish(mut self, expected_exchanges: usize) -> Result<Vec<RecordedExchange>, FixtureError> {
        self.finish_captures(expected_exchanges)
    }

    /// Stops the server and drains completed captures while retaining the target policy.
    #[expect(
        clippy::too_many_lines,
        reason = "explicit count, failure precedence, and exhaustive state conversion define the finish contract"
    )]
    fn finish_captures(&mut self, expected_exchanges: usize) -> Result<Vec<RecordedExchange>, FixtureError> {
        self.server.stop()?;
        let mut state = self.shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.excess_attempt {
            return Err(runtime_error("recording proxy exceeded its expected request count"));
        }
        if state.attempts.len() != expected_exchanges {
            return Err(runtime_error("recording proxy captured an unexpected request count"));
        }
        if state
            .attempts
            .iter()
            .any(|attempt| matches!(attempt, CaptureAttempt::FailedBeforeStart))
        {
            return Err(runtime_error("recording provider response failed before start"));
        }
        if state.attempts.iter().any(|attempt| {
            matches!(
                attempt,
                CaptureAttempt::Preparing | CaptureAttempt::Incomplete | CaptureAttempt::Pending { .. }
            )
        }) {
            return Err(runtime_error("recording provider response capture was incomplete"));
        }
        state
            .attempts
            .drain(..)
            .map(|attempt| match attempt {
                CaptureAttempt::Complete(exchange) => Ok(exchange),
                CaptureAttempt::Preparing
                | CaptureAttempt::FailedBeforeStart
                | CaptureAttempt::Incomplete
                | CaptureAttempt::Pending { .. } => {
                    Err(runtime_error("recording proxy capture accounting was inconsistent"))
                },
            })
            .collect()
    }

    /// Waits for at least `count` terminal provider attempts without missed wakeups.
    async fn wait_for_attempts(&self, count: usize, timeout: Duration) -> Result<(), FixtureError> {
        let deadline = Instant::now() + timeout;
        loop {
            let notified = self.shared.terminal.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let reached = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .attempts
                .iter()
                .filter(|attempt| !matches!(attempt, CaptureAttempt::Preparing | CaptureAttempt::Pending { .. }))
                .count()
                >= count;
            if reached {
                return Ok(());
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(runtime_error("recording proxy timed out waiting for provider attempt"));
            }
        }
    }
}

/// Records one complete scenario through a live recording proxy.
#[expect(
    clippy::large_stack_frames,
    clippy::too_many_lines,
    reason = "the explicit startup, sequential turn, join, finish, and sanitize order is the lifecycle contract"
)]
pub(super) async fn record_live(
    scenario: &InferenceScenario,
    target: ProviderTarget,
    rules: &RedactionRules,
) -> Result<WireFixture, FixtureError> {
    validate_target(&target)?;
    scenario.validate_version()?;
    validate_scenario_requests(scenario)?;
    let bound = scenario.bind_model(&target.model);
    validate_scenario_requests(&bound)?;
    validate_bound_models(&bound, &target.model)?;
    let config_source = load_and_validate_config_source(&bound)?;
    let client = build_scenario_client()?;
    let expected_turns = bound.turns.len();

    // Provider and model provenance must outlive the target moved into the
    // recorder. Credentials remain solely in the moved outbound HeaderMap.
    let provider = target.provider.clone();
    let model = target.model.clone();
    let mut recorder = RecordingProxy::start_bounded(target, expected_turns).await?;

    let listener = free_port_guard();
    let listener_port = listener.port();
    let config = match build_replay_config(
        &config_source,
        listener_port,
        &bound.upstream_authority,
        recorder.addr().port(),
    ) {
        Ok(config) => config,
        Err(error) => {
            recorder.finish(0)?;
            return Err(error);
        },
    };
    let released_port = listener.release();
    debug_assert_eq!(released_port, listener_port);
    let mut proxy = start_proxy_no_wait(&config);
    wait_for_tcp(proxy.addr());

    let scenario_id = bound.id;
    let protocol = bound.protocol;
    let mut pending = Vec::with_capacity(expected_turns);
    let mut retained_client = CaptureUsage::default();
    let client_limit = LIVE_CAPTURE_BUDGET;
    let mut operation_error = None;
    let mut attempts = 0_usize;
    let mut previous_response_id = None;
    for mut turn in bound.turns {
        turn.bind_previous_response_id(previous_response_id.as_deref())?;
        attempts = attempts.saturating_add(1);
        match send_recorded_request_with_header(
            &client,
            proxy.addr(),
            &turn.request,
            &RECORDER_CAPABILITY_HEADER,
            recorder.capability(),
        )
        .await
        {
            Ok(client_response) => {
                previous_response_id = client_response.response_id().map(str::to_owned);
                let exchange = BorrowedRecordedExchange {
                    request: &turn.request,
                    response: &client_response,
                };
                let envelope = collection_entry_overhead(pending.is_empty());
                if retain_serialized_value_with_overhead(&mut retained_client, client_limit, envelope, &exchange) {
                    pending.push(LivePendingTurn {
                        scenario: turn,
                        client_response,
                    });
                } else {
                    operation_error = Some(runtime_error("recording client capture exceeded aggregate limit"));
                }
            },
            Err(error) => operation_error = Some(error),
        }
        if let Err(error) = recorder.wait_for_attempts(attempts, RECORDING_WAIT_TIMEOUT).await {
            operation_error = Some(error);
        }
        if operation_error.is_some() {
            break;
        }
    }

    let proxy_result = proxy
        .shutdown()
        .map_err(|_error| runtime_error("scenario proxy shutdown did not complete"));
    let captured = recorder.finish_captures(attempts);
    let captured = captured?;
    proxy_result?;
    if let Some(error) = operation_error {
        return Err(error);
    }
    if pending.len() != expected_turns || captured.len() != expected_turns {
        return Err(runtime_error("recording scenario did not complete every turn"));
    }

    let mut turns = Vec::with_capacity(pending.len());
    let mut expectations = Vec::with_capacity(pending.len());
    for (pending, upstream) in pending.into_iter().zip(captured) {
        expectations.push(pending.scenario.expect);
        let turn = WireTurn {
            name: pending.scenario.name,
            client: RecordedExchange {
                request: pending.scenario.request,
                response: pending.client_response,
            },
            upstream,
        };
        turns.push(turn);
    }
    let mut fixture = WireFixture {
        version: WIRE_FIXTURE_VERSION,
        scenario_id,
        protocol,
        provenance: FixtureProvenance {
            kind: ProvenanceKind::Live,
            provider,
            model,
            source_id: None,
        },
        normalization: NormalizationMetadata {
            version: 1,
            linked_ids: BTreeMap::new(),
        },
        turns,
    };
    sanitize_fixture_preserving_structure(&mut fixture, rules)?;
    for (turn_index, (turn, expectation)) in fixture.turns.iter().zip(&expectations).enumerate() {
        validate_expectation(turn_index, turn, expectation)?;
    }
    validate_commit_safe_with_rules_and_credentials(&fixture, rules, &recorder.shared.target.outbound_headers)?;
    Ok(fixture)
}

/// Target and capture state shared by recorder connection tasks.
struct RecorderShared {
    /// Rustls Reqwest client with redirects disabled.
    client: reqwest::Client,
    /// No-proxy, redirect-disabled client that owns capability-bearing sends.
    hop_client: reqwest::Client,
    /// Ordered attempt state; no lock is held across an await.
    state: Mutex<RecorderState>,
    /// Validated provider target, including secret outbound headers.
    target: ProviderTarget,
    /// Per-run capability required before any request allocation or provider I/O.
    capability: HeaderValue,
    /// Maximum provider attempts allowed for this recorder.
    max_attempts: usize,
    /// Maximum compact and structural resources retained across exchanges.
    retained_limit: CaptureBudget,
    /// Notifies waiters after every terminal attempt transition.
    terminal: Notify,
}

/// Ordered mutable state for one recording scenario.
struct RecorderState {
    /// Provider attempts in request-arrival order.
    attempts: Vec<CaptureAttempt>,
    /// Whether an authenticated request exceeded the configured attempt budget.
    excess_attempt: bool,
    /// Compact and structural resources retained across all capture slots.
    retained: CaptureUsage,
}

/// Maximum retained resources for one live capture leg.
#[derive(Clone, Copy)]
struct CaptureBudget {
    /// Maximum compact serialized bytes.
    serialized_bytes: usize,
    /// Maximum decoded structural resources.
    structure: DocumentValidationLimits,
}

/// Resources already retained by one live capture leg.
#[derive(Clone, Copy, Default)]
struct CaptureUsage {
    /// Compact serialized bytes, including collection envelopes.
    serialized_bytes: usize,
    /// Decoded structural resources, including collection envelopes.
    structure: DocumentResourceUsage,
}

/// Lifecycle state of one provider attempt.
enum CaptureAttempt {
    /// Request entered the recorder but has not passed bounded capture.
    Preparing,
    /// Request captured while its provider response remains active.
    Pending {
        /// Captured provider-facing request.
        request: RecordedRequest,
        /// Parsed provider response retained until downstream delivery succeeds.
        response: Option<RecordedResponse>,
        /// Whether Hyper consumed the complete downstream body.
        body_delivered: bool,
        /// Whether Hyper flushed and closed the downstream HTTP/1 connection.
        connection_succeeded: bool,
    },
    /// Complete provider-originated request/response exchange.
    Complete(RecordedExchange),
    /// Provider failed before any response body byte reached downstream.
    FailedBeforeStart,
    /// Provider failed or capture became invalid after downstream response start.
    Incomplete,
}

/// Fails a reserved attempt if its handler future is cancelled before observer transfer.
struct AttemptCancellation {
    /// Shared state retained only while cancellation is armed.
    shared: Option<Arc<RecorderShared>>,
    /// Ordered attempt slot guarded by this value.
    slot: usize,
}

impl AttemptCancellation {
    /// Arms cancellation immediately after reserving one attempt slot.
    fn new(shared: Arc<RecorderShared>, slot: usize) -> Self {
        Self {
            shared: Some(shared),
            slot,
        }
    }

    /// Records one explicit terminal failure and disarms cancellation.
    fn fail(&mut self, failure: CaptureAttempt) {
        if let Some(shared) = self.shared.take() {
            fail_attempt(&shared, self.slot, failure);
        }
    }

    /// Transfers cancellation ownership to the response delivery observer.
    fn into_delivery(mut self) -> AttemptDelivery {
        AttemptDelivery {
            shared: self
                .shared
                .take()
                .expect("BUG: attempt cancellation transferred more than once"),
            slot: self.slot,
            resolved: AtomicBool::new(false),
        }
    }
}

impl Drop for AttemptCancellation {
    fn drop(&mut self) {
        self.fail(CaptureAttempt::Incomplete);
    }
}

/// Completed client leg retained until recorder accounting is joined.
struct LivePendingTurn {
    /// Bound scenario request and expectations.
    scenario: super::ScenarioTurn,
    /// Complete response returned by Praxis.
    client_response: RecordedResponse,
}

/// Borrowed exact serialization shape used for client-leg accounting.
#[derive(Serialize)]
struct BorrowedRecordedExchange<'a> {
    /// Original client request from the bound scenario.
    request: &'a RecordedRequest,
    /// Complete client response returned by Praxis.
    response: &'a RecordedResponse,
}

/// Handles one bounded JSON provider request and incrementally streams its response.
#[expect(
    clippy::too_many_lines,
    reason = "the request boundary keeps bounded capture, sanitized header projection, and outbound send order explicit"
)]
async fn handle_provider_request(request: Request<Incoming>, shared: Arc<RecorderShared>) -> Response<RecordingBody> {
    let (mut parts, incoming) = request.into_parts();
    if !take_valid_recorder_capability(&mut parts.headers, &shared.capability) {
        return opaque_json_response(StatusCode::FORBIDDEN, "recording request was not authorized");
    }
    let Some(slot) = begin_attempt(&shared) else {
        return opaque_json_response(StatusCode::TOO_MANY_REQUESTS, "recording request limit exceeded");
    };
    let mut cancellation = AttemptCancellation::new(Arc::clone(&shared), slot);
    let path = match parts.uri.path_and_query() {
        Some(path) if path.as_str().starts_with('/') => path.as_str().to_owned(),
        _ => {
            return reject_request(
                &mut cancellation,
                StatusCode::BAD_REQUEST,
                "provider request target is invalid",
            );
        },
    };
    if incoming
        .size_hint()
        .upper()
        .is_some_and(|upper| upper > u64::try_from(MAX_SCENARIO_REQUEST_BODY_BYTES).unwrap_or(u64::MAX))
    {
        return reject_request(
            &mut cancellation,
            StatusCode::PAYLOAD_TOO_LARGE,
            "provider request body exceeded recording limit",
        );
    }
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok());
    let collected = match tokio::time::timeout(
        RECORDING_REQUEST_BODY_TIMEOUT,
        Limited::new(incoming, MAX_SCENARIO_REQUEST_BODY_BYTES).collect(),
    )
    .await
    {
        Ok(Ok(collected)) => collected.to_bytes(),
        Ok(Err(_)) => {
            return reject_request(
                &mut cancellation,
                StatusCode::PAYLOAD_TOO_LARGE,
                "provider request body exceeded recording limit",
            );
        },
        Err(_) => {
            return reject_request(
                &mut cancellation,
                StatusCode::REQUEST_TIMEOUT,
                "provider request body timed out",
            );
        },
    };
    let recorded_body = match parse_request_body(content_type, &collected) {
        Ok(body @ (RecordedBody::Empty | RecordedBody::Json { .. })) => body,
        Ok(RecordedBody::Sse { .. } | RecordedBody::Base64 { .. }) | Err(_) => {
            return reject_request(
                &mut cancellation,
                StatusCode::BAD_REQUEST,
                "provider request body is not finite JSON",
            );
        },
    };
    if validate_request_body(&recorded_body).is_err() {
        return reject_request(
            &mut cancellation,
            StatusCode::PAYLOAD_TOO_LARGE,
            "provider request body exceeded recording limit",
        );
    }
    let Ok(request_headers) = http_fixture_headers(&parts.headers) else {
        return reject_request(
            &mut cancellation,
            StatusCode::BAD_REQUEST,
            "provider request headers are invalid",
        );
    };
    let Ok(outbound_headers) = provider_request_headers(&parts.headers, &shared.target.outbound_headers) else {
        return reject_request(
            &mut cancellation,
            StatusCode::BAD_REQUEST,
            "provider request headers are invalid",
        );
    };
    let provider_url = match joined_provider_url(&shared.target.base_url, &path) {
        Ok(url) => url,
        Err(error) => return reject_request(&mut cancellation, StatusCode::BAD_REQUEST, error),
    };
    let Ok(method) = reqwest::Method::from_bytes(parts.method.as_str().as_bytes()) else {
        return reject_request(
            &mut cancellation,
            StatusCode::BAD_REQUEST,
            "provider request method is invalid",
        );
    };
    let recorded_request = RecordedRequest {
        method: parts.method.as_str().to_owned(),
        path,
        headers: request_headers,
        body: recorded_body,
    };
    if !capture_request(&shared, slot, recorded_request) {
        cancellation.fail(CaptureAttempt::Incomplete);
        return opaque_json_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "provider request capture exceeded aggregate limit",
        );
    }
    let Ok(response) = shared
        .client
        .request(method, provider_url)
        .headers(outbound_headers)
        .body(collected)
        .send()
        .await
    else {
        cancellation.fail(CaptureAttempt::FailedBeforeStart);
        return opaque_json_response(StatusCode::BAD_GATEWAY, "provider response failed before start");
    };

    start_provider_response(response, shared, slot, cancellation).await
}

/// Prefetches one provider body result before committing downstream response headers.
#[expect(
    clippy::too_many_lines,
    reason = "response prefetch, header projection, empty completion, and stream handoff form one response-start boundary"
)]
async fn start_provider_response(
    response: reqwest::Response,
    shared: Arc<RecorderShared>,
    slot: usize,
    mut cancellation: AttemptCancellation,
) -> Response<RecordingBody> {
    let status = response.status();
    let Ok(recorded_headers) = http_fixture_headers(response.headers()) else {
        cancellation.fail(CaptureAttempt::FailedBeforeStart);
        return opaque_json_response(StatusCode::BAD_GATEWAY, "provider response failed before start");
    };
    let Ok(forwarded_headers) = provider_response_headers(response.headers()) else {
        cancellation.fail(CaptureAttempt::FailedBeforeStart);
        return opaque_json_response(StatusCode::BAD_GATEWAY, "provider response failed before start");
    };
    if headers_contain_configured_credential(&shared.target.outbound_headers, &forwarded_headers) {
        cancellation.fail(CaptureAttempt::Incomplete);
        return opaque_json_response(StatusCode::BAD_GATEWAY, "provider response could not be recorded");
    }
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let mut provider_stream = response.bytes_stream();
    let first = loop {
        match provider_stream.next().await {
            Some(Ok(first)) if first.is_empty() => {},
            Some(Ok(first)) if first.len() <= MAX_SCRIPTED_RESPONSE_BODY_BYTES => break Some(first),
            Some(Ok(_) | Err(_)) => {
                cancellation.fail(CaptureAttempt::FailedBeforeStart);
                return opaque_json_response(StatusCode::BAD_GATEWAY, "provider response failed before start");
            },
            None => break None,
        }
    };
    let metadata = ResponseMetadata {
        content_type,
        headers: recorded_headers,
        status: status.as_u16(),
    };
    if first.is_none() {
        if !complete_attempt(
            &shared,
            slot,
            RecordedResponse {
                status: metadata.status,
                headers: metadata.headers,
                body: RecordedBody::Empty,
            },
        ) {
            cancellation.fail(CaptureAttempt::Incomplete);
            return opaque_json_response(
                StatusCode::BAD_GATEWAY,
                "provider response capture exceeded aggregate limit",
            );
        }
        return observed_attempt_response(
            response_with_headers(status, forwarded_headers, empty_recording_body()),
            cancellation,
        );
    }

    let first = first.expect("BUG: first provider chunk was checked as present");
    let (sender, receiver) = tokio::sync::mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
    tokio::spawn(
        ForwardCapture {
            sender,
            shared: Arc::clone(&shared),
            slot,
            metadata,
        }
        .run(first, provider_stream),
    );
    let frames = stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|frame| (frame, receiver))
    });
    observed_attempt_response(
        response_with_headers(
            status,
            forwarded_headers,
            http_body_util::BodyExt::boxed(StreamBody::new(frames)),
        ),
        cancellation,
    )
}

/// Attaches delivery accounting to one provider-originated response.
fn observed_attempt_response(
    response: Response<RecordingBody>,
    cancellation: AttemptCancellation,
) -> Response<RecordingBody> {
    observe_response_delivery(response, Arc::new(cancellation.into_delivery()))
}

/// Per-attempt callbacks shared by the body wrapper and its HTTP/1 connection.
struct AttemptDelivery {
    /// Ordered recorder state.
    shared: Arc<RecorderShared>,
    /// Attempt slot whose delivery is observed.
    slot: usize,
    /// Whether a success or failure terminal transition already won.
    resolved: AtomicBool,
}

impl ResponseDelivery for AttemptDelivery {
    fn body_delivered(&self) {
        if mark_body_delivered(&self.shared, self.slot) {
            self.resolved.store(true, Ordering::Release);
        }
    }

    fn connection_succeeded(&self) {
        if mark_connection_succeeded(&self.shared, self.slot) {
            self.resolved.store(true, Ordering::Release);
        }
    }

    fn delivery_failed(&self) {
        self.fail_once();
    }
}

impl AttemptDelivery {
    /// Records at most one observer-owned failure.
    fn fail_once(&self) {
        if !self.resolved.swap(true, Ordering::AcqRel) {
            fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
        }
    }
}

impl Drop for AttemptDelivery {
    fn drop(&mut self) {
        self.fail_once();
    }
}

/// Owns the downstream channel and terminal capture transition for one stream.
struct ForwardCapture {
    /// Bounded downstream response channel.
    sender: tokio::sync::mpsc::Sender<Result<Frame<Bytes>, io::Error>>,
    /// Shared ordered recorder state.
    shared: Arc<RecorderShared>,
    /// Ordered attempt slot updated at stream completion.
    slot: usize,
    /// Provider response metadata retained without credentials.
    metadata: ResponseMetadata,
}

impl ForwardCapture {
    /// Forwards original chunks with backpressure while retaining one bounded copy.
    #[expect(
        clippy::too_many_lines,
        reason = "bounded forwarding and capture accounting remain adjacent so every stream exit records a terminal state"
    )]
    async fn run(
        self,
        first: Bytes,
        mut provider_stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    ) {
        let mut captured = BytesMut::with_capacity(first.len());
        captured.extend_from_slice(&first);
        if self.sender.send(Ok(Frame::data(first))).await.is_err() {
            fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
            return;
        }
        while let Some(chunk) = provider_stream.next().await {
            let Ok(chunk) = chunk else {
                let _sent = self
                    .sender
                    .send(Err(io::Error::other("provider response stream failed")))
                    .await;
                fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
                return;
            };
            if chunk.is_empty() {
                continue;
            }
            let Some(next_size) = captured.len().checked_add(chunk.len()) else {
                self.fail_limit().await;
                return;
            };
            if next_size > MAX_SCRIPTED_RESPONSE_BODY_BYTES {
                self.fail_limit().await;
                return;
            }
            captured.extend_from_slice(&chunk);
            if self.sender.send(Ok(Frame::data(chunk))).await.is_err() {
                fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
                return;
            }
        }

        if contains_configured_credential(&self.shared.target.outbound_headers, &captured) {
            fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
            return;
        }
        let body = match parse_response_body(self.metadata.content_type.as_deref(), &captured) {
            Ok(body) if validate_response_body(&body).is_ok() => body,
            Ok(_) | Err(_) => {
                fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
                return;
            },
        };
        if recorded_body_contains_configured_credential(&body, &self.shared.target.outbound_headers) {
            fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
            return;
        }
        complete_attempt(
            &self.shared,
            self.slot,
            RecordedResponse {
                status: self.metadata.status,
                headers: self.metadata.headers,
                body,
            },
        );
    }

    /// Sends one opaque body error and marks capture incomplete at its size ceiling.
    async fn fail_limit(&self) {
        let _sent = self
            .sender
            .send(Err(io::Error::other("provider response capture exceeded limit")))
            .await;
        fail_attempt(&self.shared, self.slot, CaptureAttempt::Incomplete);
    }
}

/// Scans parsed response text without copying payloads or formatting secrets.
fn recorded_body_contains_configured_credential(body: &RecordedBody, configured: &HeaderMap) -> bool {
    match body {
        RecordedBody::Empty => false,
        RecordedBody::Json { value } => json_contains_configured_credential(value, configured),
        RecordedBody::Sse { frames, .. } => frames.iter().any(|frame| {
            frame
                .event
                .as_deref()
                .is_some_and(|event| contains_configured_credential(configured, event.as_bytes()))
                || frame
                    .id
                    .as_deref()
                    .is_some_and(|id| contains_configured_credential(configured, id.as_bytes()))
                || contains_configured_credential(configured, frame.data.as_bytes())
                || serde_json::from_str(&frame.data)
                    .is_ok_and(|value| json_contains_configured_credential(&value, configured))
        }),
        // The bounded raw body is scanned before it is represented as Base64.
        RecordedBody::Base64 { data } => contains_configured_credential(configured, data.as_bytes()),
    }
}

/// Recursively scans borrowed JSON keys and strings, including escaped values.
fn json_contains_configured_credential(value: &serde_json::Value, configured: &HeaderMap) -> bool {
    match value {
        serde_json::Value::Array(values) => values
            .iter()
            .any(|value| json_contains_configured_credential(value, configured)),
        serde_json::Value::Object(object) => object.iter().any(|(key, value)| {
            contains_configured_credential(configured, key.as_bytes())
                || json_contains_configured_credential(value, configured)
        }),
        serde_json::Value::String(text) => contains_configured_credential(configured, text.as_bytes()),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => false,
    }
}

/// Provider response metadata retained without credential headers.
struct ResponseMetadata {
    /// Parsed content type used only to select the recorded body representation.
    content_type: Option<String>,
    /// Fixture-safe ordered response header values.
    headers: BTreeMap<String, Vec<String>>,
    /// Provider response status.
    status: u16,
}

/// Validates all target fields before any loopback listener is bound.
fn validate_target(target: &ProviderTarget) -> Result<(), FixtureError> {
    let valid_transport = match target.base_url.scheme() {
        "https" => true,
        "http" => target
            .base_url
            .host()
            .is_some_and(|host| is_literal_loopback_host(&host) || is_credentialless_private_vllm_http(target, &host)),
        _ => false,
    };
    let valid_url = valid_transport
        && target.base_url.has_host()
        && target.base_url.username().is_empty()
        && target.base_url.password().is_none()
        && target.base_url.query().is_none()
        && target.base_url.fragment().is_none()
        && !target.base_url.cannot_be_a_base();
    if target.provider.trim().is_empty() || target.model.trim().is_empty() || !valid_url {
        return Err(runtime_error("recording provider target is invalid"));
    }
    validate_provider_headers(&target.outbound_headers)
}

/// Accepts only literal IPv4 and IPv6 loopback hosts without DNS resolution.
fn is_literal_loopback_host(host: &Host<&str>) -> bool {
    match host {
        Host::Domain(_) => false,
        Host::Ipv4(address) => Ipv4Addr::is_loopback(address),
        Host::Ipv6(address) => Ipv6Addr::is_loopback(address),
    }
}

/// Accepts only credentialless vLLM targets addressed by literal private IP.
fn is_credentialless_private_vllm_http(target: &ProviderTarget, host: &Host<&str>) -> bool {
    target.provider == "vllm"
        && target.outbound_headers.is_empty()
        && match host {
            Host::Domain(_) => false,
            Host::Ipv4(address) => Ipv4Addr::is_private(address),
            Host::Ipv6(address) => Ipv6Addr::is_unique_local(address),
        }
}

/// Generates one high-entropy ASCII capability without exposing it in errors.
fn generate_recorder_capability() -> Result<HeaderValue, FixtureError> {
    let encoded = STANDARD.encode(rand::random::<[u8; 32]>());
    let mut capability =
        HeaderValue::from_str(&encoded).map_err(|_source| runtime_error("recording capability generation failed"))?;
    capability.set_sensitive(true);
    Ok(capability)
}

/// Builds a redirect-free Reqwest client pinned to the Rustls backend.
fn build_provider_client(root: Option<reqwest::Certificate>) -> Result<reqwest::Client, FixtureError> {
    build_provider_client_with_timeout(root, PROVIDER_REQUEST_TIMEOUT)
}

/// Builds one provider client with an injected request deadline for deterministic tests.
fn build_provider_client_with_timeout(
    root: Option<reqwest::Certificate>,
    request_timeout: Duration,
) -> Result<reqwest::Client, FixtureError> {
    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .use_rustls_tls()
        .connect_timeout(Duration::from_secs(10))
        .timeout(request_timeout)
        .redirect(reqwest::redirect::Policy::none());
    if let Some(root) = root {
        builder = builder.add_root_certificate(root);
    }
    builder
        .build()
        .map_err(|_source| runtime_error("recording provider client could not be built"))
}

/// Builds the capability-bearing client with proxies and redirects disabled.
fn build_recorder_hop_client() -> Result<reqwest::Client, FixtureError> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_source| runtime_error("recording hop client could not be built"))
}

/// Builds the single redirect-free client used for all scenario turns.
fn build_scenario_client() -> Result<reqwest::Client, FixtureError> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_source| FixtureError::ReplayHttp)
}

/// Appends one exact inbound origin-form path/query to the configured base path.
fn joined_provider_url(base: &reqwest::Url, origin: &str) -> Result<reqwest::Url, &'static str> {
    if !origin.starts_with('/') || origin.starts_with("//") || origin.contains('#') {
        return Err("provider request target is invalid");
    }
    let origin_path = origin.split_once('?').map_or(origin, |(path, _query)| path);
    if origin_path.split('/').any(is_dot_path_segment) {
        return Err("provider request target is invalid");
    }
    let base_serialized = base.as_str().strip_suffix('/').unwrap_or(base.as_str());
    let expected_base_path = base.path().strip_suffix('/').unwrap_or_else(|| base.path());
    let expected_path = format!("{expected_base_path}{origin_path}");
    let joined = reqwest::Url::parse(&format!("{base_serialized}{origin}"))
        .map_err(|_source| "provider request target is invalid")?;
    let expected_query = origin.split_once('?').map(|(_path, query)| query);
    let same_authority = joined.scheme() == base.scheme()
        && joined.username() == base.username()
        && joined.password() == base.password()
        && joined.host_str() == base.host_str()
        && joined.port() == base.port()
        && joined.port_or_known_default() == base.port_or_known_default();
    if !same_authority
        || joined.path() != expected_path
        || joined.query() != expected_query
        || joined.fragment().is_some()
    {
        return Err("provider request target is invalid");
    }
    Ok(joined)
}

/// Detects exact raw, encoded, or mixed `.` and `..` path segments without allocation.
fn is_dot_path_segment(segment: &str) -> bool {
    let bytes = segment.as_bytes();
    let mut index = 0_usize;
    let mut dots = 0_usize;
    while index < bytes.len() {
        if bytes[index] == b'.' {
            index = index.saturating_add(1);
        } else if bytes.get(index..index.saturating_add(3)).is_some_and(|encoded| {
            encoded.len() == 3 && encoded[0] == b'%' && encoded[1] == b'2' && encoded[2].eq_ignore_ascii_case(&b'e')
        }) {
            index = index.saturating_add(3);
        } else {
            return false;
        }
        dots = dots.saturating_add(1);
    }
    matches!(dots, 1 | 2)
}

/// Requires the protocol request's top-level provider-selection model to match.
fn validate_bound_models(scenario: &InferenceScenario, model: &str) -> Result<(), FixtureError> {
    for turn in &scenario.turns {
        let RecordedBody::Json { value } = &turn.request.body else {
            return Err(runtime_error("recording scenario model does not match provider target"));
        };
        if value
            .as_object()
            .and_then(|request| request.get("model"))
            .and_then(serde_json::Value::as_str)
            != Some(model)
        {
            return Err(runtime_error("recording scenario model does not match provider target"));
        }
    }
    Ok(())
}

/// Requires exactly one matching capability and strips it before all projections.
fn take_valid_recorder_capability(headers: &mut HeaderMap, expected: &HeaderValue) -> bool {
    let mut values = headers.get_all(&RECORDER_CAPABILITY_HEADER).iter();
    let valid = values.next().is_some_and(|value| value == expected) && values.next().is_none();
    headers.remove(&RECORDER_CAPABILITY_HEADER);
    valid
}

/// Reserves an ordered slot after capability authentication.
fn begin_attempt(shared: &RecorderShared) -> Option<usize> {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    if state.attempts.len() >= shared.max_attempts {
        state.excess_attempt = true;
        return None;
    }
    let slot = state.attempts.len();
    state.attempts.push(CaptureAttempt::Preparing);
    drop(state);
    Some(slot)
}

/// Attaches a validated, bounded request before starting provider I/O.
fn capture_request(shared: &RecorderShared, slot: usize, request: RecordedRequest) -> bool {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    if !matches!(state.attempts.get(slot), Some(CaptureAttempt::Preparing)) {
        return false;
    }
    let envelope = recorded_exchange_entry_overhead(state.retained.serialized_bytes == 0);
    if !retain_serialized_value_with_overhead(&mut state.retained, shared.retained_limit, envelope, &request) {
        state.attempts[slot] = CaptureAttempt::Incomplete;
        drop(state);
        shared.terminal.notify_waiters();
        return false;
    }
    if let Some(attempt @ CaptureAttempt::Preparing) = state.attempts.get_mut(slot) {
        *attempt = CaptureAttempt::Pending {
            request,
            response: None,
            body_delivered: false,
            connection_succeeded: false,
        };
        true
    } else {
        false
    }
}

/// Stores one provider-originated response until both delivery boundaries succeed.
fn complete_attempt(shared: &RecorderShared, slot: usize, response: RecordedResponse) -> bool {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    if !matches!(state.attempts.get(slot), Some(CaptureAttempt::Pending { .. })) {
        return false;
    }
    if !retain_serialized_value_with_overhead(
        &mut state.retained,
        shared.retained_limit,
        CaptureUsage::default(),
        &response,
    ) {
        state.attempts[slot] = CaptureAttempt::Incomplete;
        drop(state);
        shared.terminal.notify_waiters();
        return false;
    }
    let completed = state.attempts.get_mut(slot).is_some_and(|attempt| {
        if let CaptureAttempt::Pending {
            response: pending_response,
            ..
        } = attempt
        {
            *pending_response = Some(response);
        }
        finish_delivered_attempt(attempt)
    });
    drop(state);
    if completed {
        shared.terminal.notify_waiters();
    }
    true
}

/// Accounts for one retained value and its exact collection envelope.
fn retain_serialized_value_with_overhead<T: Serialize>(
    retained: &mut CaptureUsage,
    maximum: CaptureBudget,
    envelope: CaptureUsage,
    value: &T,
) -> bool {
    let Some(bytes_with_envelope) = retained.serialized_bytes.checked_add(envelope.serialized_bytes) else {
        return false;
    };
    let Some(remaining) = maximum.serialized_bytes.checked_sub(bytes_with_envelope) else {
        return false;
    };
    let Some(structure_with_envelope) =
        add_structure_with_limit(retained.structure, envelope.structure, maximum.structure)
    else {
        return false;
    };
    let Some(structural_limits) = remaining_structure(maximum.structure, structure_with_envelope) else {
        return false;
    };
    let Ok((size, structure)) = measure_json_value_with_limits(value, remaining, structural_limits) else {
        return false;
    };
    retained.serialized_bytes = bytes_with_envelope.saturating_add(size);
    retained.structure = add_structure(structure_with_envelope, structure);
    true
}

/// Returns the exact compact-JSON overhead added by one array entry.
const fn collection_entry_overhead(first: bool) -> CaptureUsage {
    CaptureUsage {
        serialized_bytes: if first { 2 } else { 1 },
        structure: DocumentResourceUsage {
            nodes: if first { 1 } else { 0 },
            container_entries: 1,
            decoded_string_bytes: 0,
            max_depth: 0,
        },
    }
}

/// Returns exchange object plus surrounding collection overhead.
const fn recorded_exchange_entry_overhead(first: bool) -> CaptureUsage {
    let collection = collection_entry_overhead(first);
    CaptureUsage {
        serialized_bytes: RECORDED_EXCHANGE_ENVELOPE_BYTES + collection.serialized_bytes,
        structure: DocumentResourceUsage {
            nodes: collection.structure.nodes + 1,
            container_entries: collection.structure.container_entries + 2,
            decoded_string_bytes: b"request".len() + b"response".len(),
            max_depth: 0,
        },
    }
}

/// Adds aggregate structural usage while refusing overflow or the first excess.
fn add_structure_with_limit(
    current: DocumentResourceUsage,
    next: DocumentResourceUsage,
    maximum: DocumentValidationLimits,
) -> Option<DocumentResourceUsage> {
    let combined = DocumentResourceUsage {
        nodes: current.nodes.checked_add(next.nodes)?,
        container_entries: current.container_entries.checked_add(next.container_entries)?,
        decoded_string_bytes: current.decoded_string_bytes.checked_add(next.decoded_string_bytes)?,
        max_depth: current.max_depth.max(next.max_depth),
    };
    (combined.nodes <= maximum.max_nodes
        && combined.container_entries <= maximum.max_container_entries
        && combined.decoded_string_bytes <= maximum.max_decoded_string_bytes
        && combined.max_depth <= maximum.max_depth)
        .then_some(combined)
}

/// Returns one aggregate structural remainder while depth remains per-value.
fn remaining_structure(
    maximum: DocumentValidationLimits,
    retained: DocumentResourceUsage,
) -> Option<DocumentValidationLimits> {
    Some(DocumentValidationLimits {
        max_nodes: maximum.max_nodes.checked_sub(retained.nodes)?,
        max_container_entries: maximum.max_container_entries.checked_sub(retained.container_entries)?,
        max_decoded_string_bytes: maximum
            .max_decoded_string_bytes
            .checked_sub(retained.decoded_string_bytes)?,
        max_depth: maximum.max_depth,
    })
}

/// Adds one already-validated resource cost without wrapping on hostile input.
fn add_structure(current: DocumentResourceUsage, next: DocumentResourceUsage) -> DocumentResourceUsage {
    DocumentResourceUsage {
        nodes: current.nodes.saturating_add(next.nodes),
        container_entries: current.container_entries.saturating_add(next.container_entries),
        decoded_string_bytes: current.decoded_string_bytes.saturating_add(next.decoded_string_bytes),
        max_depth: current.max_depth.max(next.max_depth),
    }
}


/// Marks one body EOF and completes the attempt if the connection also succeeded.
fn mark_body_delivered(shared: &RecorderShared, slot: usize) -> bool {
    mark_delivery_boundary(shared, slot, |body_delivered, _connection_succeeded| {
        *body_delivered = true;
    })
}

/// Marks one clean HTTP/1 connection finish and completes after body EOF.
fn mark_connection_succeeded(shared: &RecorderShared, slot: usize) -> bool {
    mark_delivery_boundary(shared, slot, |_body_delivered, connection_succeeded| {
        *connection_succeeded = true;
    })
}

/// Updates one delivery flag without holding recorder state across an await.
fn mark_delivery_boundary(shared: &RecorderShared, slot: usize, update: impl FnOnce(&mut bool, &mut bool)) -> bool {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    let completed = state.attempts.get_mut(slot).is_some_and(|attempt| {
        if let CaptureAttempt::Pending {
            body_delivered,
            connection_succeeded,
            ..
        } = attempt
        {
            update(body_delivered, connection_succeeded);
        }
        finish_delivered_attempt(attempt)
    });
    drop(state);
    if completed {
        shared.terminal.notify_waiters();
    }
    completed
}

/// Atomically converts a fully parsed and delivered attempt to `Complete`.
fn finish_delivered_attempt(attempt: &mut CaptureAttempt) -> bool {
    let ready = matches!(
        attempt,
        CaptureAttempt::Pending {
            response: Some(_),
            body_delivered: true,
            connection_succeeded: true,
            ..
        }
    );
    if !ready {
        return false;
    }
    let CaptureAttempt::Pending {
        request,
        response: Some(response),
        ..
    } = std::mem::replace(attempt, CaptureAttempt::Incomplete)
    else {
        return false;
    };
    *attempt = CaptureAttempt::Complete(RecordedExchange { request, response });
    true
}

/// Transitions one pending slot to a deterministic failure state.
fn fail_attempt(shared: &RecorderShared, slot: usize, failure: CaptureAttempt) -> bool {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    let transitioned = if let Some(attempt) = state.attempts.get_mut(slot)
        && matches!(attempt, CaptureAttempt::Preparing | CaptureAttempt::Pending { .. })
    {
        *attempt = failure;
        true
    } else {
        false
    };
    drop(state);
    if transitioned {
        shared.terminal.notify_waiters();
    }
    transitioned
}

/// Rejects one request while preserving terminal lifecycle accounting.
fn reject_request(
    cancellation: &mut AttemptCancellation,
    status: StatusCode,
    message: &'static str,
) -> Response<RecordingBody> {
    cancellation.fail(CaptureAttempt::Incomplete);
    opaque_json_response(status, message)
}

/// Builds a provider response with sanitized transport headers.
fn response_with_headers(status: StatusCode, headers: HeaderMap, body: RecordingBody) -> Response<RecordingBody> {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// Builds one static opaque JSON response with no provider or credential detail.
fn opaque_json_response(status: StatusCode, message: &'static str) -> Response<RecordingBody> {
    let body = Bytes::from(format!(r#"{{"error":"{message}"}}"#));
    let mut response = Response::new(full_recording_body(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

/// Builds an empty response body for readiness and empty provider responses.
fn empty_recording_body() -> RecordingBody {
    full_recording_body(Bytes::new())
}

/// Converts an infallible full body into the recorder's error-capable body type.
fn full_recording_body(bytes: Bytes) -> RecordingBody {
    Full::new(bytes).map_err(|never| match never {}).boxed()
}

/// Creates an opaque recorder error with no target, fixture, or credential data.
fn runtime_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use std::{
        collections::BTreeMap,
        convert::Infallible,
        error::Error as _,
        fmt::Write as _,
        future::Future,
        io,
        net::{IpAddr, Ipv4Addr, SocketAddr},
        pin::Pin,
        process::{Command, ExitStatus, Stdio},
        sync::{
            Arc, Mutex as StdMutex,
            atomic::{AtomicBool as StdAtomicBool, AtomicUsize, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use bytes::{Bytes, BytesMut};
    use futures::{StreamExt as _, stream};
    use http::{HeaderMap, HeaderName, HeaderValue, Request, Response, StatusCode, header};
    use http_body_util::{BodyExt as _, Full, StreamBody, combinators::BoxBody};
    use hyper::{
        body::{Frame, Incoming},
        server::conn::http1,
        service::service_fn,
    };
    use hyper_util::rt::TokioIo;
    use serde_json::{Value, json};
    use tokio::{
        io::{AsyncReadExt as _, AsyncWriteExt as _},
        net::{TcpListener, TcpStream},
        sync::{Notify, mpsc, oneshot},
        task::{JoinHandle, JoinSet},
    };
    use tokio_rustls::TlsAcceptor;

    use super::{
        BorrowedRecordedExchange, CaptureAttempt, CaptureBudget, CaptureUsage, ForwardCapture, LIVE_CAPTURE_BUDGET,
        LIVE_CAPTURE_STRUCTURE_LIMITS, MAX_INFERENCE_TURNS, ProviderTarget, RECORDER_CAPABILITY_HEADER, RecorderShared,
        RecorderState, RecordingProxy, ResponseMetadata, begin_attempt, build_provider_client,
        build_provider_client_with_timeout, capture_request, collection_entry_overhead, complete_attempt, fail_attempt,
        joined_provider_url, recorded_body_contains_configured_credential, retain_serialized_value_with_overhead,
        validate_bound_models, validate_target,
    };
    use crate::{
        inference_fixture::{
            BodyKind, FixtureError, InferenceProtocol, InferenceScenario, RecordedBody, RecordedExchange,
            RecordedRequest, RecordedResponse, ScenarioExpectation, ScenarioRunner, ScenarioTurn,
        },
        net::tls::TestCertificates,
    };

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    const SECRET_SENTINEL: &str = "task-six-secret-sentinel";
    const NO_PROXY_CHILD_ENV: &str = "PRAXIS_INFERENCE_NO_PROXY_CHILD_PROVIDER_URL";
    const NO_PROXY_CHILD_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);
    const NO_PROXY_CHILD_DEADLINE: Duration = Duration::from_secs(2);
    const CHILD_STATUS_POLL_INTERVAL: Duration = Duration::from_millis(5);
    const NO_PROXY_TEST_NAME: &str = "inference_fixture::record::tests::provider_client_ignores_ambient_proxy_settings";
    const STALLED_CHILD_ENV: &str = "PRAXIS_INFERENCE_STALLED_CHILD_READY_PATH";
    const STALLED_CHILD_TEST_NAME: &str =
        "inference_fixture::record::tests::child_deadline_kills_reaps_and_runs_cleanup";

    type ProviderBody = BoxBody<Bytes, io::Error>;
    type HandlerFuture = Pin<Box<dyn Future<Output = Response<ProviderBody>> + Send>>;
    type Handler = Arc<dyn Fn(Request<Incoming>) -> HandlerFuture + Send + Sync>;

    /// Verifies that a joined test server no longer owns its listener address.
    async fn assert_listener_released(addr: SocketAddr) {
        let deadline = tokio::time::Instant::now() + TEST_TIMEOUT;
        loop {
            match std::net::TcpListener::bind(addr) {
                Ok(listener) => {
                    drop(listener);
                    return;
                },
                Err(error) if error.kind() == io::ErrorKind::AddrInUse && tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                },
                Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
                    panic!("test listener address remained in use after bounded cleanup");
                },
                Err(error) => panic!("test listener address could not be rebound: {error}"),
            }
        }
    }

    #[tokio::test]
    async fn listener_release_check_tolerates_brief_reuse_by_another_test() {
        // Catches an instantaneous rebind assertion racing a concurrently reused ephemeral port.
        let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let releaser = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            drop(listener);
        });

        assert_listener_released(addr).await;

        releaser.await.unwrap();
    }

    enum ChildOutcome {
        Exited(ExitStatus),
        TimedOut(ExitStatus),
    }

    fn run_child_with_deadline(
        command: &mut Command,
        timeout: Duration,
        cleanup: impl FnOnce(),
    ) -> io::Result<ChildOutcome> {
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let result = (|| {
            let deadline = Instant::now()
                .checked_add(timeout)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "child deadline overflow"))?;
            let mut child = command.spawn()?;
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => return Ok(ChildOutcome::Exited(status)),
                    Ok(None) => {},
                    Err(error) => {
                        let _kill_result = child.kill();
                        let _wait_result = child.wait();
                        return Err(error);
                    },
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    let kill_result = child.kill();
                    let wait_result = child.wait();
                    kill_result?;
                    return wait_result.map(ChildOutcome::TimedOut);
                }
                thread::sleep(remaining.min(CHILD_STATUS_POLL_INTERVAL));
            }
        })();
        cleanup();
        result
    }

    #[test]
    fn child_deadline_kills_reaps_and_runs_cleanup() {
        if let Some(ready_path) = std::env::var_os(STALLED_CHILD_ENV) {
            use std::io::Write as _;

            let output = [b'x'; 128 * 1024];
            io::stdout().lock().write_all(&output).unwrap();
            io::stderr().lock().write_all(&output).unwrap();
            std::fs::write(ready_path, b"ready").unwrap();
            thread::park();
            unreachable!("stalled child must be terminated by its parent");
        }

        let ready_dir = tempfile::tempdir().unwrap();
        let ready_path = ready_dir.path().join("ready");
        let cleanup_ran = Arc::new(StdAtomicBool::new(false));
        let cleanup_flag = Arc::clone(&cleanup_ran);
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", STALLED_CHILD_TEST_NAME, "--nocapture", "--test-threads=1"])
            .env(STALLED_CHILD_ENV, &ready_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let outcome = run_child_with_deadline(&mut command, Duration::from_secs(1), move || {
            cleanup_flag.store(true, Ordering::Release);
        })
        .unwrap();

        assert!(matches!(outcome, ChildOutcome::TimedOut(status) if !status.success()));
        assert!(
            ready_path.is_file(),
            "the child must enter its deliberate stall before termination"
        );
        assert!(
            cleanup_ran.load(Ordering::Acquire),
            "cleanup must run after timeout and reap"
        );
    }

    #[tokio::test]
    async fn provider_client_ignores_ambient_proxy_settings() {
        if let Some(provider_url) = std::env::var_os(NO_PROXY_CHILD_ENV) {
            let sentinel = STANDARD.encode(rand::random::<[u8; 32]>());
            let mut credential = HeaderValue::from_str(&sentinel).unwrap();
            credential.set_sensitive(true);
            let response = build_provider_client(None)
                .expect("provider client should build")
                .get(provider_url.to_str().expect("test provider URL should be UTF-8"))
                .timeout(NO_PROXY_CHILD_REQUEST_TIMEOUT)
                .header(HeaderName::from_static("x-api-key"), credential)
                .send()
                .await
                .expect("synthetic credential request should reach the intended provider");
            assert!(
                response.status().is_success(),
                "intended provider should respond successfully"
            );
            return;
        }

        let provider_listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let provider_addr = provider_listener.local_addr().unwrap();
        let proxy_listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        let stop = Arc::new(StdAtomicBool::new(false));
        let provider_connected = Arc::new(StdAtomicBool::new(false));
        let provider_saw_credential = Arc::new(StdAtomicBool::new(false));
        let proxy_connected = Arc::new(StdAtomicBool::new(false));

        let provider_task = spawn_connection_observer(
            provider_listener,
            Arc::clone(&stop),
            Arc::clone(&provider_connected),
            Some(Arc::clone(&provider_saw_credential)),
            b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
        let proxy_task = spawn_connection_observer(
            proxy_listener,
            Arc::clone(&stop),
            Arc::clone(&proxy_connected),
            None,
            b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );

        let proxy_url = format!("http://{proxy_addr}");
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", NO_PROXY_TEST_NAME, "--nocapture", "--test-threads=1"])
            .env(NO_PROXY_CHILD_ENV, format!("http://{provider_addr}/credential-check"))
            .env("HTTP_PROXY", &proxy_url)
            .env("HTTPS_PROXY", &proxy_url)
            .env("ALL_PROXY", &proxy_url)
            .env("http_proxy", &proxy_url)
            .env("https_proxy", &proxy_url)
            .env("all_proxy", &proxy_url)
            .env_remove("NO_PROXY")
            .env_remove("no_proxy")
            .env_remove("ANTHROPIC_API_KEY")
            .env_remove("OPENAI_API_KEY")
            .env_remove("INFERENCE_PROVIDER_API_KEY");
        let outcome = run_child_with_deadline(&mut command, NO_PROXY_CHILD_DEADLINE, move || {
            stop.store(true, Ordering::Release);
            provider_task.join().unwrap();
            proxy_task.join().unwrap();
        })
        .expect("isolated test child should start and remain supervised");
        let ChildOutcome::Exited(status) = outcome else {
            panic!("isolated test child exceeded its hard deadline");
        };

        assert!(
            status.success(),
            "isolated test child should complete a direct provider request"
        );
        assert!(
            provider_connected.load(Ordering::Acquire),
            "intended loopback provider should receive the request"
        );
        assert!(
            provider_saw_credential.load(Ordering::Acquire),
            "intended loopback provider should receive the synthetic credential header"
        );
        assert!(
            !proxy_connected.load(Ordering::Acquire),
            "ambient proxy listener must not observe a provider connection"
        );
    }

    fn spawn_connection_observer(
        listener: std::net::TcpListener,
        stop: Arc<StdAtomicBool>,
        connected: Arc<StdAtomicBool>,
        saw_credential: Option<Arc<StdAtomicBool>>,
        response: &'static [u8],
    ) -> thread::JoinHandle<()> {
        listener.set_nonblocking(true).unwrap();
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _peer)) => {
                        connected.store(true, Ordering::Release);
                        if let Some(saw_credential) = saw_credential {
                            use std::io::{Read as _, Write as _};

                            stream.set_nonblocking(false).unwrap();
                            stream.set_read_timeout(Some(NO_PROXY_CHILD_REQUEST_TIMEOUT)).unwrap();
                            let mut request = [0_u8; 4096];
                            let mut bytes_read = 0;
                            while bytes_read < request.len()
                                && !request[..bytes_read].windows(4).any(|window| window == b"\r\n\r\n")
                            {
                                let read = stream.read(&mut request[bytes_read..]).unwrap_or(0);
                                if read == 0 {
                                    break;
                                }
                                bytes_read += read;
                            }
                            saw_credential.store(
                                request[..bytes_read]
                                    .windows(b"x-api-key:".len())
                                    .any(|window| window.eq_ignore_ascii_case(b"x-api-key:")),
                                Ordering::Release,
                            );
                            stream.write_all(response).unwrap();
                        } else {
                            use std::io::Write as _;

                            stream.write_all(response).unwrap();
                        }
                        return;
                    },
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(1)),
                    Err(error) => panic!("loopback observer failed: {error}"),
                }
            }
        })
    }

    #[test]
    fn parsed_response_scan_finds_credentials_in_json_keys_and_sse_metadata() {
        let secret = r#"structured-\"credential\"\tail"#;
        let configured = credential_headers(header::AUTHORIZATION, &format!("Bearer {secret}"));
        let mut keyed = serde_json::Map::new();
        keyed.insert(secret.to_owned(), json!("safe value"));
        let nested_sse_json = serde_json::to_string(&json!({
            "outer": {"inner": secret},
        }))
        .unwrap();
        let bodies = [
            (
                "JSON key",
                RecordedBody::Json {
                    value: Value::Object(keyed),
                },
            ),
            (
                "SSE event",
                RecordedBody::Sse {
                    frames: vec![super::super::SseFrame {
                        event: Some(secret.to_owned()),
                        data: "safe".to_owned(),
                        id: None,
                        retry: None,
                    }],
                    done: false,
                },
            ),
            (
                "SSE id",
                RecordedBody::Sse {
                    frames: vec![super::super::SseFrame {
                        event: None,
                        data: "safe".to_owned(),
                        id: Some(secret.to_owned()),
                        retry: None,
                    }],
                    done: false,
                },
            ),
            (
                "nested escaped SSE JSON",
                RecordedBody::Sse {
                    frames: vec![super::super::SseFrame {
                        event: None,
                        data: nested_sse_json,
                        id: None,
                        retry: None,
                    }],
                    done: false,
                },
            ),
        ];

        for (case, body) in bodies {
            assert!(
                recorded_body_contains_configured_credential(&body, &configured),
                "{case} must be rejected"
            );
        }
    }

    #[test]
    fn provider_eof_alone_keeps_attempt_pending_until_downstream_delivery() {
        let shared = isolated_recorder_shared();
        let slot = begin_attempt(&shared).unwrap();
        capture_request(
            &shared,
            slot,
            RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
        );
        complete_attempt(
            &shared,
            slot,
            RecordedResponse {
                status: 204,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
        );

        assert!(matches!(
            shared.state.lock().unwrap().attempts.as_slice(),
            [CaptureAttempt::Pending { .. }]
        ));
    }

    #[test]
    fn recorder_capture_budget_rejects_the_first_value_past_the_remaining_limit() {
        let mut shared = isolated_recorder_shared();
        let request = RecordedRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let response = RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let exact_exchange_bytes = serde_json::to_vec(&vec![RecordedExchange {
            request: request.clone(),
            response: response.clone(),
        }])
        .unwrap()
        .len();
        shared.retained_limit.serialized_bytes = exact_exchange_bytes - 1;

        let slot = begin_attempt(&shared).unwrap();
        assert!(capture_request(&shared, slot, request));
        assert!(!complete_attempt(&shared, slot, response));
        assert!(matches!(
            shared.state.lock().unwrap().attempts.as_slice(),
            [CaptureAttempt::Incomplete]
        ));
    }

    #[test]
    fn recorder_capture_budget_counts_the_exact_exchange_collection_envelope() {
        let mut shared = isolated_recorder_shared();
        let request = RecordedRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let response = RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let exact_exchange_bytes = serde_json::to_vec(&vec![RecordedExchange {
            request: request.clone(),
            response: response.clone(),
        }])
        .unwrap()
        .len();
        shared.retained_limit.serialized_bytes = exact_exchange_bytes;

        let slot = begin_attempt(&shared).unwrap();
        assert!(capture_request(&shared, slot, request));
        assert!(complete_attempt(&shared, slot, response));

        assert_eq!(
            shared.state.lock().unwrap().retained.serialized_bytes,
            exact_exchange_bytes
        );
    }

    #[test]
    fn recorder_capture_budget_counts_the_first_retained_exchange_after_an_early_failure() {
        let mut shared = isolated_recorder_shared();
        let request = RecordedRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let response = RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let exact_exchange_bytes = serde_json::to_vec(&vec![RecordedExchange {
            request: request.clone(),
            response: response.clone(),
        }])
        .unwrap()
        .len();
        shared.retained_limit.serialized_bytes = exact_exchange_bytes;

        let failed_slot = begin_attempt(&shared).unwrap();
        fail_attempt(&shared, failed_slot, CaptureAttempt::FailedBeforeStart);
        let retained_slot = begin_attempt(&shared).unwrap();
        assert!(capture_request(&shared, retained_slot, request));
        assert!(complete_attempt(&shared, retained_slot, response));

        assert_eq!(
            shared.state.lock().unwrap().retained.serialized_bytes,
            exact_exchange_bytes
        );
    }

    #[test]
    fn client_capture_budget_counts_the_exact_exchange_collection_envelope() {
        let request = RecordedRequest {
            method: "POST".to_owned(),
            path: "/v1/responses".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let response = RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };
        let exact_exchange_bytes = serde_json::to_vec(&vec![RecordedExchange {
            request: request.clone(),
            response: response.clone(),
        }])
        .unwrap()
        .len();
        let borrowed = BorrowedRecordedExchange {
            request: &request,
            response: &response,
        };
        let envelope = collection_entry_overhead(true);
        let mut retained = CaptureUsage::default();

        assert!(retain_serialized_value_with_overhead(
            &mut retained,
            CaptureBudget {
                serialized_bytes: exact_exchange_bytes,
                structure: LIVE_CAPTURE_STRUCTURE_LIMITS,
            },
            envelope,
            &borrowed,
        ));
        assert_eq!(retained.serialized_bytes, exact_exchange_bytes);
    }

    #[test]
    fn recorder_capture_budget_rejects_structure_before_retaining_the_request() {
        let mut shared = isolated_recorder_shared();
        shared.retained_limit.structure.max_nodes = 1;
        let slot = begin_attempt(&shared).unwrap();

        assert!(!capture_request(
            &shared,
            slot,
            RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
        ));
        assert!(matches!(
            shared.state.lock().unwrap().attempts.as_slice(),
            [CaptureAttempt::Incomplete]
        ));
    }

    #[test]
    fn provider_url_rejects_dot_segments_and_preserves_safe_origin_identity() {
        let base = UrlCase::parse("http://provider.test/api//");
        for unsafe_origin in [
            "/../escape",
            "/./same",
            "/%2e/encoded",
            "/%2E/case",
            "/.%2e/mixed",
            "/%2e./mixed",
            "/%2e%2e/encoded",
            "/%2e%2E/encoded-case",
            "//attacker.invalid/authority-like",
            "http://attacker.invalid/absolute",
        ] {
            assert!(
                joined_provider_url(&base, unsafe_origin).is_err(),
                "unsafe origin target must be rejected: {unsafe_origin}"
            );
        }

        let joined = joined_provider_url(&base, "/v1/%66oo?literal=%2e&slash=%2F").unwrap();
        assert_eq!(
            joined.as_str(),
            "http://provider.test/api//v1/%66oo?literal=%2e&slash=%2F"
        );
    }

    #[test]
    fn model_validation_is_required_at_top_level_and_ignores_nested_model_keys() {
        for protocol in [
            InferenceProtocol::OpenaiResponses,
            InferenceProtocol::AnthropicMessages,
            InferenceProtocol::OpenaiChatCompletions,
        ] {
            let mut scenario = live_scenario().bind_model("fixture-model");
            scenario.protocol = protocol;
            {
                let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
                    panic!("live scenario request should be JSON");
                };
                value["tools"] = json!([{
                    "model": 42,
                    "schema": {"model": {"arbitrary": true}}
                }]);
                value["metadata"] = json!({"model": "nested-other-model"});
            }
            validate_bound_models(&scenario, "fixture-model")
                .expect("nested model keys are provider payload data, not selection fields");

            for invalid_top_level in [Value::Null, json!(42), json!("other-model")] {
                let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
                    panic!("live scenario request should be JSON");
                };
                value["model"] = invalid_top_level;
                assert!(validate_bound_models(&scenario, "fixture-model").is_err());
            }
            let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
                panic!("live scenario request should be JSON");
            };
            value.as_object_mut().unwrap().remove("model");
            assert!(validate_bound_models(&scenario, "fixture-model").is_err());
        }
    }

    #[tokio::test]
    async fn forward_capture_skips_many_empty_provider_chunks_without_queueing_frames() {
        let shared = Arc::new(isolated_recorder_shared());
        let slot = begin_attempt(&shared).unwrap();
        capture_request(
            &shared,
            slot,
            RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
        );
        let (sender, mut receiver) = mpsc::channel(1);
        let capture = ForwardCapture {
            sender,
            shared,
            slot,
            metadata: ResponseMetadata {
                content_type: Some("application/octet-stream".to_owned()),
                headers: BTreeMap::new(),
                status: 200,
            },
        };
        let chunks = std::iter::repeat_n(Bytes::new(), 10_000)
            .chain([Bytes::from_static(b"b")])
            .map(Ok::<_, reqwest::Error>);
        let forward = capture.run(Bytes::from_static(b"a"), stream::iter(chunks));
        let drain = async {
            let mut frames = Vec::new();
            while let Some(frame) = receiver.recv().await {
                frames.push(frame.unwrap().into_data().unwrap());
            }
            frames
        };

        let ((), frames) = tokio::join!(forward, drain);
        assert_eq!(frames, [Bytes::from_static(b"a"), Bytes::from_static(b"b")]);
    }

    #[tokio::test]
    async fn forward_capture_releases_provider_chunk_allocations_before_eof() {
        struct TrackedByte {
            byte: [u8; 1],
            dropped: Arc<AtomicUsize>,
        }

        impl AsRef<[u8]> for TrackedByte {
            fn as_ref(&self) -> &[u8] {
                &self.byte
            }
        }

        impl Drop for TrackedByte {
            fn drop(&mut self) {
                self.dropped.fetch_add(1, Ordering::SeqCst);
            }
        }

        const CHUNK_COUNT: usize = 64;
        let shared = Arc::new(isolated_recorder_shared());
        let slot = begin_attempt(&shared).unwrap();
        capture_request(
            &shared,
            slot,
            RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/chat/completions".to_owned(),
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
        );
        let (sender, mut receiver) = mpsc::channel(1);
        let capture = ForwardCapture {
            sender,
            shared: Arc::clone(&shared),
            slot,
            metadata: ResponseMetadata {
                content_type: Some("application/octet-stream".to_owned()),
                headers: BTreeMap::new(),
                status: 200,
            },
        };
        let dropped = Arc::new(AtomicUsize::new(0));
        let (provider_tx, provider_rx) = futures::channel::mpsc::unbounded();
        for _ in 0..CHUNK_COUNT {
            provider_tx
                .unbounded_send(Ok(Bytes::from_owner(TrackedByte {
                    byte: [b'b'],
                    dropped: Arc::clone(&dropped),
                })))
                .unwrap();
        }
        let capture_task = tokio::spawn(capture.run(Bytes::from_static(b"a"), provider_rx));

        for _ in 0..=CHUNK_COUNT {
            let frame = receiver.recv().await.unwrap().unwrap();
            drop(frame.into_data().unwrap());
        }
        assert_eq!(dropped.load(Ordering::SeqCst), CHUNK_COUNT);

        drop(provider_tx);
        capture_task.await.unwrap();
        let mut expected = Vec::with_capacity(CHUNK_COUNT + 1);
        expected.push(b'a');
        expected.extend(std::iter::repeat_n(b'b', CHUNK_COUNT));
        let expected = STANDARD.encode(expected);
        let captured_exact_body = {
            let state = shared.state.lock().unwrap();
            matches!(
                state.attempts.as_slice(),
                [CaptureAttempt::Pending {
                    response: Some(RecordedResponse {
                        body: RecordedBody::Base64 { data },
                        ..
                    }),
                    ..
                }] if data == &expected
            )
        };
        assert!(
            captured_exact_body,
            "capture must complete with the exact provider body"
        );
    }

    #[tokio::test]
    async fn json_proxy_preserves_origin_path_body_safe_multi_headers_and_never_captures_credentials() {
        let (observed_tx, observed_rx) = oneshot::channel();
        let observed_tx = Arc::new(std::sync::Mutex::new(Some(observed_tx)));
        let provider = LocalProvider::start(handler(move |request| {
            let observed_tx = Arc::clone(&observed_tx);
            async move {
                let (parts, body) = request.into_parts();
                let body = body.collect().await.unwrap().to_bytes();
                observed_tx.lock().unwrap().take().unwrap().send((parts, body)).unwrap();

                let mut response = json_response(StatusCode::CREATED, &json!({"ok": true}));
                response
                    .headers_mut()
                    .append("request-id", HeaderValue::from_static("provider-a"));
                response
                    .headers_mut()
                    .append("request-id", HeaderValue::from_static("provider-b"));
                response.headers_mut().insert(
                    header::AUTHORIZATION,
                    HeaderValue::from_static("Bearer provider-response-secret"),
                );
                response
            }
        }))
        .await;
        let mut outbound_headers = HeaderMap::new();
        outbound_headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {SECRET_SENTINEL}")).unwrap(),
        );
        outbound_headers.append("openai-beta", HeaderValue::from_static("first"));
        outbound_headers.append("openai-beta", HeaderValue::from_static("second"));
        let target = target(provider.http_url("/api%2Fprefix/"), outbound_headers);
        let recorder = RecordingProxy::start(target).await.unwrap();
        let client = authorized_client(&recorder);

        let response = client
            .post(format!(
                "http://{}/v1/%2fitem?encoded=%2F&space=a+b&repeat=1&repeat=2",
                recorder.addr()
            ))
            .header(header::AUTHORIZATION, "Bearer inbound-secret")
            .header("openai-beta", "inbound-must-be-replaced")
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"prompt":"hello"}"#)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        assert!(!response.headers().contains_key(header::AUTHORIZATION));
        assert_eq!(
            response
                .headers()
                .get_all("request-id")
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect::<Vec<_>>(),
            ["provider-a", "provider-b"]
        );
        assert_eq!(response.json::<Value>().await.unwrap(), json!({"ok": true}));

        let (parts, provider_body) = observed_rx.await.unwrap();
        assert_eq!(
            parts.uri.path_and_query().unwrap().as_str(),
            "/api%2Fprefix/v1/%2fitem?encoded=%2F&space=a+b&repeat=1&repeat=2"
        );
        assert_eq!(provider_body, Bytes::from_static(br#"{"prompt":"hello"}"#));
        assert!(!parts.headers.contains_key(&RECORDER_CAPABILITY_HEADER));
        assert!(
            parts
                .headers
                .get(header::AUTHORIZATION)
                .is_some_and(|value| value == format!("Bearer {SECRET_SENTINEL}").as_str()),
            "provider should receive the configured authorization value"
        );
        assert_eq!(
            parts
                .headers
                .get_all("openai-beta")
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );

        let exchanges = recorder.finish(1).unwrap();
        let serialized = serde_json::to_string(&exchanges).unwrap();
        assert!(!serialized.contains(SECRET_SENTINEL));
        assert!(!serialized.contains("inbound-secret"));
        assert!(!serialized.contains("provider-response-secret"));
        assert_eq!(
            exchanges[0].request.path,
            "/v1/%2fitem?encoded=%2F&space=a+b&repeat=1&repeat=2"
        );
        assert_eq!(
            exchanges[0].response.headers.get("request-id"),
            Some(&vec!["provider-a".to_owned(), "provider-b".to_owned()])
        );
        provider.finish().await;
    }

    #[tokio::test]
    async fn invalid_recorder_capabilities_never_reach_the_provider_or_attempt_state() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&provider_hits);
        let provider = LocalProvider::start(handler(move |_request| {
            let handler_hits = Arc::clone(&handler_hits);
            async move {
                handler_hits.fetch_add(1, Ordering::SeqCst);
                json_response(StatusCode::OK, &json!({"unexpected": true}))
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), secret_headers()))
            .await
            .unwrap();

        let client = test_client();
        let missing = client
            .post(format!("http://{}/v1/responses", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model"}"#)
            .send()
            .await
            .unwrap();
        let wrong = client
            .post(format!("http://{}/v1/responses", recorder.addr()))
            .header(&RECORDER_CAPABILITY_HEADER, "wrong-capability")
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model"}"#)
            .send()
            .await
            .unwrap();
        let duplicate = client
            .post(format!("http://{}/v1/responses", recorder.addr()))
            .header(&RECORDER_CAPABILITY_HEADER, recorder.capability())
            .header(&RECORDER_CAPABILITY_HEADER, recorder.capability())
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model"}"#)
            .send()
            .await
            .unwrap();
        let readiness = client.get(format!("http://{}/", recorder.addr())).send().await.unwrap();

        assert_eq!(missing.status(), StatusCode::FORBIDDEN);
        assert_eq!(wrong.status(), StatusCode::FORBIDDEN);
        assert_eq!(duplicate.status(), StatusCode::FORBIDDEN);
        assert_eq!(readiness.status(), StatusCode::FORBIDDEN);
        assert_eq!(provider_hits.load(Ordering::SeqCst), 0);
        assert!(recorder.finish(0).unwrap().is_empty());
        provider.finish().await;
    }

    #[tokio::test]
    async fn recorder_authorization_rejects_requests_for_other_destinations() {
        let recorder = RecordingProxy::start_bounded(target(UrlCase::parse("http://127.0.0.1/"), HeaderMap::new()), 1)
            .await
            .unwrap();
        let foreign_request = test_client().post("http://127.0.0.1:1/v1/responses").build().unwrap();

        let error = recorder.send(foreign_request).await.unwrap_err();

        assert_eq!(error.to_string(), "recording request destination is invalid");
        assert!(recorder.finish(0).unwrap().is_empty());
    }

    #[tokio::test]
    async fn recorder_send_never_follows_a_provider_redirect() {
        let redirect_hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&redirect_hits);
        let redirect_target = LocalProvider::start(handler(move |_request| {
            let handler_hits = Arc::clone(&handler_hits);
            async move {
                handler_hits.fetch_add(1, Ordering::SeqCst);
                json_response(StatusCode::OK, &json!({"unexpected": true}))
            }
        }))
        .await;
        let location = redirect_target.http_url("/capability-must-not-arrive");
        let provider = LocalProvider::start(handler(move |_request| {
            let location = location.clone();
            async move {
                let mut response = Response::new(infallible_body(Bytes::new()));
                *response.status_mut() = StatusCode::FOUND;
                response
                    .headers_mut()
                    .insert(header::LOCATION, HeaderValue::from_str(location.as_str()).unwrap());
                response
            }
        }))
        .await;
        let recorder = RecordingProxy::start_bounded(target(provider.http_url("/"), secret_headers()), 1)
            .await
            .unwrap();
        let request = test_client()
            .post(format!("http://{}/v1/responses", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model"}"#)
            .build()
            .unwrap();

        let response = recorder.send(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FOUND);
        let _body = response.bytes().await.unwrap();

        assert_eq!(redirect_hits.load(Ordering::SeqCst), 0);
        assert_eq!(recorder.finish(1).unwrap().len(), 1);
        provider.finish().await;
        redirect_target.finish().await;
    }

    #[tokio::test]
    async fn authenticated_requests_cannot_exceed_the_recorder_attempt_budget() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&provider_hits);
        let provider = LocalProvider::start(handler(move |request| {
            let handler_hits = Arc::clone(&handler_hits);
            async move {
                assert!(!request.headers().contains_key(&RECORDER_CAPABILITY_HEADER));
                handler_hits.fetch_add(1, Ordering::SeqCst);
                json_response(StatusCode::OK, &json!({"ok": true}))
            }
        }))
        .await;
        let recorder = RecordingProxy::start_bounded(target(provider.http_url("/"), secret_headers()), 1)
            .await
            .unwrap();
        let client = test_client();

        assert!(!format!("{:?}", recorder.capability()).contains(recorder.capability().to_str().unwrap()));
        let first = recorder
            .send(
                client
                    .post(format!("http://{}/v1/responses", recorder.addr()))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(r#"{"model":"fixture-model"}"#)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let _body = first.bytes().await.unwrap();
        let second = recorder
            .send(
                client
                    .post(format!("http://{}/v1/responses", recorder.addr()))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(r#"{"model":"fixture-model"}"#)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(provider_hits.load(Ordering::SeqCst), 1);
        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording proxy exceeded its expected request count");
        provider.finish().await;
    }

    #[tokio::test]
    async fn stalled_headers_release_the_single_recorder_connection_slot() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&provider_hits);
        let provider = LocalProvider::start(handler(move |_request| {
            let handler_hits = Arc::clone(&handler_hits);
            async move {
                handler_hits.fetch_add(1, Ordering::SeqCst);
                json_response(StatusCode::OK, &json!({"ok": true}))
            }
        }))
        .await;
        let recorder = RecordingProxy::start_bounded(target(provider.http_url("/"), secret_headers()), 1)
            .await
            .unwrap();
        let mut stalled = TcpStream::connect(recorder.addr()).await.unwrap();
        let partial = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: recorder\r\n{}: {}\r\n",
            RECORDER_CAPABILITY_HEADER,
            recorder.capability().to_str().unwrap()
        );
        stalled.write_all(partial.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let response = tokio::time::timeout(
            TEST_TIMEOUT,
            authorized_client(&recorder)
                .post(format!("http://{}/v1/responses", recorder.addr()))
                .header(header::CONTENT_TYPE, "application/json")
                .body(r#"{"model":"fixture-model"}"#)
                .send(),
        )
        .await
        .expect("the header timeout must release the bounded connection slot")
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.bytes().await.unwrap();
        assert_eq!(provider_hits.load(Ordering::SeqCst), 1);
        drop(stalled);
        assert_eq!(recorder.finish(1).unwrap().len(), 1);
        provider.finish().await;
    }

    #[tokio::test]
    async fn stalled_request_body_times_out_and_releases_the_recorder_connection_slot() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&provider_hits);
        let provider = LocalProvider::start(handler(move |_request| {
            let handler_hits = Arc::clone(&handler_hits);
            async move {
                handler_hits.fetch_add(1, Ordering::SeqCst);
                json_response(StatusCode::OK, &json!({"ok": true}))
            }
        }))
        .await;
        let recorder = RecordingProxy::start_bounded(target(provider.http_url("/"), secret_headers()), 2)
            .await
            .unwrap();
        let mut stalled = TcpStream::connect(recorder.addr()).await.unwrap();
        let partial = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: recorder\r\n{}: {}\r\nContent-Type: application/json\r\nContent-Length: 128\r\n\r\n{{",
            RECORDER_CAPABILITY_HEADER,
            recorder.capability().to_str().unwrap()
        );
        stalled.write_all(partial.as_bytes()).await.unwrap();

        let response = tokio::time::timeout(
            super::RECORDING_REQUEST_BODY_TIMEOUT + Duration::from_secs(2),
            authorized_client(&recorder)
                .post(format!("http://{}/v1/responses", recorder.addr()))
                .header(header::CONTENT_TYPE, "application/json")
                .body(r#"{"model":"fixture-model"}"#)
                .send(),
        )
        .await
        .expect("the body timeout must release the bounded connection slot")
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.bytes().await.unwrap();
        assert_eq!(provider_hits.load(Ordering::SeqCst), 1);
        drop(stalled);
        assert!(recorder.finish(2).is_err());
        provider.finish().await;
    }

    #[tokio::test]
    async fn streaming_proxy_forwards_first_frame_before_provider_release_and_records_logical_sse() {
        let (provider_waiting_tx, provider_waiting_rx) = oneshot::channel();
        let provider_waiting_tx = Arc::new(std::sync::Mutex::new(Some(provider_waiting_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(std::sync::Mutex::new(Some(release_rx)));
        let provider = LocalProvider::start(handler(move |_request| {
            let provider_waiting_tx = Arc::clone(&provider_waiting_tx);
            let release_rx = Arc::clone(&release_rx);
            async move {
                let chunks = vec![
                    Bytes::from_static(b"data: {\"part\":"),
                    Bytes::from_static(b"1}\n\n"),
                    Bytes::from_static(b"event: second\ndata: {\"part\":2}\n\ndata: {\"part\":3}\n\ndata: [DO"),
                    Bytes::from_static(b"NE]\n\n"),
                ];
                let state = StreamState {
                    chunks: chunks.into(),
                    index: 0,
                    provider_waiting: provider_waiting_tx.lock().unwrap().take(),
                    release: release_rx.lock().unwrap().take(),
                };
                let body = stream::unfold(state, |mut state| async move {
                    if state.index == 2 {
                        if let Some(waiting) = state.provider_waiting.take() {
                            let _sent = waiting.send(());
                        }
                        if let Some(release) = state.release.take() {
                            let _released = release.await;
                        }
                    }
                    let chunk = state.chunks.pop_front()?;
                    state.index += 1;
                    Some((Ok::<_, io::Error>(Frame::data(chunk)), state))
                });
                let mut response = Response::new(http_body_util::BodyExt::boxed(StreamBody::new(body)));
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                response
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), HeaderMap::new()))
            .await
            .unwrap();
        let client = authorized_client(&recorder);

        let response = client
            .post(format!("http://{}/v1/chat/completions", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"stream":true}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.bytes_stream();
        let mut first_frame = BytesMut::new();
        while !first_frame.windows(2).any(|window| window == b"\n\n") {
            first_frame.extend_from_slice(&stream.next().await.unwrap().unwrap());
        }
        tokio::time::timeout(TEST_TIMEOUT, provider_waiting_rx)
            .await
            .expect("provider should reach its causal gate")
            .unwrap();
        assert_eq!(&*first_frame, b"data: {\"part\":1}\n\n");
        release_tx.send(()).unwrap();
        while let Some(chunk) = stream.next().await {
            first_frame.extend_from_slice(&chunk.unwrap());
        }

        let exchanges = recorder.finish(1).unwrap();
        let RecordedBody::Sse { frames, done } = &exchanges[0].response.body else {
            panic!("provider stream should be captured as logical SSE");
        };
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].event, None);
        assert_eq!(frames[0].data, r#"{"part":1}"#);
        assert_eq!(frames[1].event.as_deref(), Some("second"));
        assert_eq!(frames[1].data, r#"{"part":2}"#);
        assert_eq!(frames[2].data, r#"{"part":3}"#);
        assert!(done);
        provider.finish().await;
    }

    #[tokio::test]
    async fn absolute_form_authority_and_redirects_never_change_the_configured_target() {
        let trap_hits = Arc::new(AtomicUsize::new(0));
        let trap = LocalProvider::start(handler({
            let trap_hits = Arc::clone(&trap_hits);
            move |_request| {
                let trap_hits = Arc::clone(&trap_hits);
                async move {
                    trap_hits.fetch_add(1, Ordering::SeqCst);
                    json_response(StatusCode::OK, &json!({"trap": true}))
                }
            }
        }))
        .await;
        let (path_tx, mut path_rx) = mpsc::channel(2);
        let provider = LocalProvider::start(handler({
            let path_tx = path_tx.clone();
            let trap_url = trap.http_url("/stolen");
            move |request| {
                let path_tx = path_tx.clone();
                let trap_url = trap_url.clone();
                async move {
                    path_tx.send(request.uri().to_string()).await.unwrap();
                    let mut response = empty_response(StatusCode::FOUND);
                    response
                        .headers_mut()
                        .insert(header::LOCATION, HeaderValue::from_str(trap_url.as_str()).unwrap());
                    response
                }
            }
        }))
        .await;
        drop(path_tx);
        let recorder = RecordingProxy::start(target(provider.http_url("/prefix/"), HeaderMap::new()))
            .await
            .unwrap();

        let raw = authorized_raw_request(
            &recorder,
            "POST http://attacker.invalid/encoded/%2F?q=%2f HTTP/1.1\r\nHost: attacker.invalid\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        )
        .await;
        assert!(raw.starts_with("HTTP/1.1 302"));
        assert_eq!(path_rx.recv().await.as_deref(), Some("/prefix/encoded/%2F?q=%2f"));
        assert_eq!(trap_hits.load(Ordering::SeqCst), 0);

        let exchanges = recorder.finish(1).unwrap();
        assert_eq!(exchanges[0].request.path, "/encoded/%2F?q=%2f");
        provider.finish().await;
        trap.finish().await;
    }

    #[tokio::test]
    async fn raw_dot_segment_variants_and_network_path_never_reach_provider() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider = LocalProvider::start(handler({
            let provider_hits = Arc::clone(&provider_hits);
            move |_request| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::SeqCst);
                    json_response(StatusCode::OK, &json!({"unexpected": true}))
                }
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/api//"), HeaderMap::new()))
            .await
            .unwrap();
        let unsafe_targets = [
            "/../escape",
            "/./same",
            "/%2e/encoded",
            "/%2E/case",
            "/.%2e/mixed",
            "/%2e./mixed",
            "/%2e%2e/encoded",
            "//attacker.invalid/network-path",
        ];

        for target in unsafe_targets {
            let request = format!(
                "POST {target} HTTP/1.1\r\nHost: recorder\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            );
            let raw = authorized_raw_request(&recorder, &request).await;
            assert!(
                raw.starts_with("HTTP/1.1 400"),
                "unsafe target was not rejected: {target}"
            );
        }

        assert_eq!(provider_hits.load(Ordering::SeqCst), 0);
        let error = recorder.finish(unsafe_targets.len()).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        provider.finish().await;
    }

    #[tokio::test]
    async fn invalid_targets_fail_before_listener_start_without_exposing_header_values() {
        let mut invalid = vec![
            target(UrlCase::parse("ftp://127.0.0.1/api"), HeaderMap::new()),
            target(UrlCase::parse("http://user:pass@127.0.0.1/api"), HeaderMap::new()),
            target(UrlCase::parse("http://127.0.0.1/api?secret=query"), HeaderMap::new()),
            target(UrlCase::parse("http://127.0.0.1/api#fragment"), HeaderMap::new()),
        ];
        let mut empty_provider = target(UrlCase::parse("http://127.0.0.1/"), HeaderMap::new());
        empty_provider.provider.clear();
        invalid.push(empty_provider);
        let mut empty_model = target(UrlCase::parse("http://127.0.0.1/"), HeaderMap::new());
        empty_model.model.clear();
        invalid.push(empty_model);
        for unsafe_header in [header::HOST, header::CONNECTION, header::CONTENT_LENGTH] {
            let mut headers = HeaderMap::new();
            headers.insert(unsafe_header, HeaderValue::from_static(SECRET_SENTINEL));
            invalid.push(target(UrlCase::parse("http://127.0.0.1/"), headers));
        }
        for target in invalid {
            let error = RecordingProxy::start(target).await.unwrap_err();
            assert_eq!(error.to_string(), "recording provider target is invalid");
            assert_secret_absent_from_error(&error);
        }

        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, HeaderValue::from_static(SECRET_SENTINEL));
        let target = target(UrlCase::parse("http://127.0.0.1/"), headers);
        let debug = format!("{target:?}");
        assert!(!debug.contains(SECRET_SENTINEL));
    }

    #[test]
    fn provider_target_transport_accepts_only_safe_https_loopback_and_private_vllm_targets() {
        for (url, provider, credentialed, accepted) in [
            ("https://api.example.com", "compatible", false, true),
            ("http://127.0.0.1", "compatible", true, true),
            ("http://[::1]", "compatible", false, true),
            ("http://10.0.0.99:8000", "vllm", false, true),
            ("http://172.16.0.1", "vllm", false, true),
            ("http://192.168.1.1", "vllm", false, true),
            ("http://[fd00::1]", "vllm", false, true),
            ("http://10.0.0.99", "vllm", true, false),
            ("http://10.0.0.99", "compatible", false, false),
            ("http://8.8.8.8", "vllm", false, false),
            ("http://vllm.internal", "vllm", false, false),
            ("http://localhost", "vllm", false, false),
        ] {
            let headers = if credentialed {
                secret_headers()
            } else {
                HeaderMap::new()
            };
            let mut provider_target = target(UrlCase::parse(url), headers);
            provider_target.provider = provider.to_owned();

            let result = validate_target(&provider_target);

            if accepted {
                result.expect("safe recording target must be accepted");
            } else {
                let error = result.expect_err("unsafe recording target must be rejected");
                assert_eq!(error.to_string(), "recording provider target is invalid");
                assert_secret_absent_from_error(&error);
            }
        }
    }

    #[tokio::test]
    async fn provider_client_times_out_when_response_headers_never_arrive() {
        // Catches removing the provider request deadline and leaving a recorder task stuck on a stalled provider.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let provider = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _read = stream.read(&mut request).await;
            std::future::pending::<()>().await;
        });
        let client = build_provider_client_with_timeout(None, Duration::from_millis(100)).unwrap();

        let outcome = tokio::time::timeout(
            Duration::from_millis(250),
            client.get(format!("http://{addr}/stalled")).send(),
        )
        .await;
        provider.abort();
        let _stopped = provider.await;

        let error = outcome
            .expect("the provider client must enforce its own timeout")
            .expect_err("a provider that never responds must time out");
        assert!(error.is_timeout());
    }

    #[tokio::test]
    async fn provider_client_times_out_when_response_body_stalls() {
        // Catches a deadline that stops at response headers and leaves body capture blocked indefinitely.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let provider = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _read = stream.read(&mut request).await;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
                .await
                .unwrap();
            std::future::pending::<()>().await;
        });
        let client = build_provider_client_with_timeout(None, Duration::from_millis(100)).unwrap();
        let response = client.get(format!("http://{addr}/stalled")).send().await.unwrap();

        let outcome = tokio::time::timeout(Duration::from_millis(250), response.bytes()).await;
        provider.abort();
        let _stopped = provider.await;

        let error = outcome
            .expect("the provider client must retain its timeout while streaming the body")
            .expect_err("a provider body that never completes must time out");
        assert!(error.is_timeout());
    }

    #[tokio::test]
    async fn malformed_request_is_terminally_accounted_and_never_becomes_a_fixture() {
        let recorder = RecordingProxy::start(target(UrlCase::parse("http://127.0.0.1:9/"), secret_headers()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();

        let response = authorized_client(&recorder)
            .post(format!("http://{recorder_addr}/v1/chat/completions"))
            .header(header::CONTENT_TYPE, "application/json")
            .body("{")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_secret_absent_from_error(&error);
        assert_listener_released(recorder_addr).await;
    }

    #[tokio::test]
    async fn reqwest_rustls_client_reaches_repository_test_certificate_provider() {
        let certificates = TestCertificates::generate();
        let provider = TlsProvider::start(&certificates).await;
        let certificate = reqwest::Certificate::from_der(&certificates.ca_cert_der).unwrap();
        let target = target(provider.url("/tls-prefix/"), HeaderMap::new());
        let recorder = RecordingProxy::start_with_test_root(target, certificate).await.unwrap();

        let response = authorized_client(&recorder)
            .post(format!("http://{}/v1/probe", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.json::<Value>().await.unwrap(), json!({"tls": "rustls"}));
        let exchanges = recorder.finish(1).unwrap();
        assert_eq!(exchanges[0].request.path, "/v1/probe");
        provider.finish().await;
    }

    #[tokio::test]
    async fn disconnect_before_body_start_returns_opaque_502_and_refuses_a_completed_capture() {
        let provider = DisconnectProvider::start_before_headers().await;
        let recorder = RecordingProxy::start(target(provider.url(), secret_headers()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();

        let response = authorized_client(&recorder)
            .post(format!("http://{recorder_addr}/v1/chat/completions"))
            .header(header::CONTENT_TYPE, "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            json!({"error": "provider response failed before start"})
        );
        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response failed before start");
        assert_secret_absent_from_error(&error);
        assert_listener_released(recorder_addr).await;
        provider.finish().await;
    }

    #[tokio::test]
    async fn disconnect_after_body_start_forwards_partial_bytes_and_reports_incomplete_capture() {
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(std::sync::Mutex::new(Some(release_rx)));
        let provider = LocalProvider::start(handler(move |_request| {
            let release_rx = Arc::clone(&release_rx);
            async move {
                let (sender, receiver) = mpsc::channel(1);
                let release = release_rx.lock().unwrap().take().unwrap();
                tokio::spawn(async move {
                    sender
                        .send(Ok::<_, io::Error>(Frame::data(Bytes::from_static(
                            b"data: {\"partial\":true}\n\n",
                        ))))
                        .await
                        .unwrap();
                    let _released = release.await;
                    let _sent = sender.send(Err(io::Error::other("test provider disconnect"))).await;
                });
                let frames = stream::unfold(receiver, |mut receiver| async move {
                    receiver.recv().await.map(|frame| (frame, receiver))
                });
                let mut response = Response::new(http_body_util::BodyExt::boxed(StreamBody::new(frames)));
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                response
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), secret_headers()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();
        let response = authorized_client(&recorder)
            .post(format!("http://{recorder_addr}/v1/chat/completions"))
            .header(header::CONTENT_TYPE, "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.bytes_stream();
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            Bytes::from_static(b"data: {\"partial\":true}\n\n")
        );
        release_tx.send(()).unwrap();
        assert!(stream.next().await.unwrap().is_err());

        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_secret_absent_from_error(&error);
        assert_listener_released(recorder_addr).await;
        provider.finish().await;
    }

    #[tokio::test]
    async fn invalid_utf8_sse_parse_error_refuses_capture_and_releases_listener() {
        let provider = LocalProvider::start(handler(move |_request| async move {
            let mut response = Response::new(infallible_body(Bytes::from_static(&[0xFF])));
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            response
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), HeaderMap::new()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();

        let response = authorized_client(&recorder)
            .post(format!("http://{recorder_addr}/v1/chat/completions"))
            .header(header::CONTENT_TYPE, "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(response.bytes().await.unwrap(), Bytes::from_static(&[0xFF]));

        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_listener_released(recorder_addr).await;
        provider.finish().await;
    }

    #[tokio::test]
    async fn handler_cancellation_before_response_observer_is_terminal_before_provider_release() {
        let (provider_entered_tx, provider_entered_rx) = oneshot::channel();
        let provider_entered_tx = Arc::new(StdMutex::new(Some(provider_entered_tx)));
        let (release_provider_tx, release_provider_rx) = oneshot::channel();
        let release_provider_rx = Arc::new(StdMutex::new(Some(release_provider_rx)));
        let provider = LocalProvider::start(handler(move |_request| {
            let provider_entered_tx = Arc::clone(&provider_entered_tx);
            let release_provider_rx = Arc::clone(&release_provider_rx);
            async move {
                provider_entered_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                let release = release_provider_rx.lock().unwrap().take().unwrap();
                let _released = release.await;
                empty_response(StatusCode::NO_CONTENT)
            }
        }))
        .await;
        let connection_finished = Arc::new(Notify::new());
        let recorder = RecordingProxy::start_with_connection_finished(
            target(provider.http_url("/"), HeaderMap::new()),
            Arc::clone(&connection_finished),
        )
        .await
        .unwrap();
        let recorder_addr = recorder.addr();
        let mut client = TcpStream::connect(recorder_addr).await.unwrap();
        let request = with_recorder_capability(
            &recorder,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: recorder\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
        );
        client.write_all(request.as_bytes()).await.unwrap();
        provider_entered_rx.await.unwrap();
        {
            let connection_exit = connection_finished.notified();
            tokio::pin!(connection_exit);
            connection_exit.as_mut().enable();
            let terminal = recorder.shared.terminal.notified();
            tokio::pin!(terminal);
            terminal.as_mut().enable();

            drop(client);
            tokio::time::timeout(TEST_TIMEOUT, connection_exit)
                .await
                .expect("connection task should exit while provider headers remain blocked");
            tokio::time::timeout(Duration::from_secs(1), terminal)
                .await
                .expect("cancelled handler should notify a terminal attempt before provider release");
            assert!(matches!(
                recorder.shared.state.lock().unwrap().attempts.as_slice(),
                [CaptureAttempt::Incomplete]
            ));
        }

        release_provider_tx.send(()).unwrap();
        provider.finish().await;
        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_listener_released(recorder_addr).await;
    }

    #[tokio::test]
    async fn downstream_disconnect_before_empty_response_delivery_refuses_capture() {
        let (observed_tx, observed_rx) = oneshot::channel();
        let observed_tx = Arc::new(std::sync::Mutex::new(Some(observed_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(std::sync::Mutex::new(Some(release_rx)));
        let provider = LocalProvider::start(handler(move |_request| {
            let observed_tx = Arc::clone(&observed_tx);
            let release_rx = Arc::clone(&release_rx);
            async move {
                observed_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                let release = release_rx.lock().unwrap().take().unwrap();
                let _released = release.await;
                empty_response(StatusCode::NO_CONTENT)
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), HeaderMap::new()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();
        let mut client = TcpStream::connect(recorder_addr).await.unwrap();
        let request = with_recorder_capability(
            &recorder,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: recorder\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
        );
        client.write_all(request.as_bytes()).await.unwrap();
        observed_rx.await.unwrap();
        drop(client);
        release_tx.send(()).unwrap();
        provider.finish().await;

        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_listener_released(recorder_addr).await;
    }

    #[tokio::test]
    async fn downstream_disconnect_before_small_response_delivery_refuses_capture() {
        let (observed_tx, observed_rx) = oneshot::channel();
        let observed_tx = Arc::new(std::sync::Mutex::new(Some(observed_tx)));
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(std::sync::Mutex::new(Some(release_rx)));
        let provider = LocalProvider::start(handler(move |_request| {
            let observed_tx = Arc::clone(&observed_tx);
            let release_rx = Arc::clone(&release_rx);
            async move {
                observed_tx.lock().unwrap().take().unwrap().send(()).unwrap();
                let release = release_rx.lock().unwrap().take().unwrap();
                let _released = release.await;
                json_response(StatusCode::OK, &json!({"small": true}))
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), HeaderMap::new()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();
        let mut client = TcpStream::connect(recorder_addr).await.unwrap();
        let request = with_recorder_capability(
            &recorder,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: recorder\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
        );
        client.write_all(request.as_bytes()).await.unwrap();
        observed_rx.await.unwrap();
        drop(client);
        release_tx.send(()).unwrap();
        provider.finish().await;

        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_listener_released(recorder_addr).await;
    }

    #[tokio::test]
    async fn downstream_disconnect_after_first_stream_chunk_refuses_capture() {
        let (release_tx, release_rx) = oneshot::channel();
        let release_rx = Arc::new(std::sync::Mutex::new(Some(release_rx)));
        let provider = LocalProvider::start(handler(move |_request| {
            let release_rx = Arc::clone(&release_rx);
            async move {
                let (sender, receiver) = mpsc::channel(1);
                let release = release_rx.lock().unwrap().take().unwrap();
                tokio::spawn(async move {
                    sender
                        .send(Ok::<_, io::Error>(Frame::data(Bytes::from_static(
                            b"data: {\"first\":true}\n\n",
                        ))))
                        .await
                        .unwrap();
                    let _released = release.await;
                    let _sent = sender
                        .send(Ok(Frame::data(Bytes::from_static(
                            b"data: {\"second\":true}\n\ndata: [DONE]\n\n",
                        ))))
                        .await;
                });
                let frames = stream::unfold(receiver, |mut receiver| async move {
                    receiver.recv().await.map(|frame| (frame, receiver))
                });
                let mut response = Response::new(http_body_util::BodyExt::boxed(StreamBody::new(frames)));
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                response
            }
        }))
        .await;
        let recorder = RecordingProxy::start(target(provider.http_url("/"), HeaderMap::new()))
            .await
            .unwrap();
        let recorder_addr = recorder.addr();
        let mut client = TcpStream::connect(recorder_addr).await.unwrap();
        let request = with_recorder_capability(
            &recorder,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: recorder\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
        );
        client.write_all(request.as_bytes()).await.unwrap();
        let mut received = Vec::new();
        while !received
            .windows(b"data: {\"first\":true}\n\n".len())
            .any(|window| window == b"data: {\"first\":true}\n\n")
        {
            let mut buffer = [0_u8; 256];
            let read = client.read(&mut buffer).await.unwrap();
            assert_ne!(read, 0, "response must deliver its first stream chunk");
            received.extend_from_slice(&buffer[..read]);
        }
        drop(client);
        release_tx.send(()).unwrap();
        provider.finish().await;

        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_listener_released(recorder_addr).await;
    }

    #[tokio::test]
    async fn record_live_uses_one_pipeline_for_two_turns_sanitizes_once_and_is_commit_safe() {
        let request_count = Arc::new(AtomicUsize::new(0));
        let provider = LocalProvider::start(handler({
            let request_count = Arc::clone(&request_count);
            move |request| {
                let request_count = Arc::clone(&request_count);
                async move {
                    assert!(
                        request
                            .headers()
                            .get(header::AUTHORIZATION)
                            .is_some_and(|value| value == format!("Bearer {SECRET_SENTINEL}").as_str()),
                        "provider should receive configured authorization"
                    );
                    let index = request_count.fetch_add(1, Ordering::SeqCst);
                    chat_response(index)
                }
            }
        }))
        .await;
        let scenario = live_scenario();
        let target = target(provider.http_url("/"), secret_headers());

        let fixture = ScenarioRunner::record_live(&scenario, target).await.unwrap();

        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        assert_eq!(fixture.turns.len(), 2);
        assert_eq!(fixture.provenance.model, "fixture-model");
        assert_eq!(fixture.provenance.provider, "local-provider");
        assert_eq!(fixture.turns[0].upstream.request.path, "/v1/chat/completions");
        assert_eq!(fixture.turns[1].upstream.request.path, "/v1/chat/completions");
        let serialized = serde_json::to_string(&fixture).unwrap();
        assert!(!serialized.contains(SECRET_SENTINEL));
        super::super::validate_commit_safe(&fixture).unwrap();
        provider.finish().await;
    }

    #[tokio::test]
    async fn record_live_binds_previous_response_id_across_responses_turns() {
        let request_count = Arc::new(AtomicUsize::new(0));
        let provider = LocalProvider::start(handler({
            let request_count = Arc::clone(&request_count);
            move |_request| {
                let request_count = Arc::clone(&request_count);
                async move {
                    let index = request_count.fetch_add(1, Ordering::SeqCst);
                    chat_response(index)
                }
            }
        }))
        .await;
        let scenario = live_responses_scenario();
        let target = target(provider.http_url("/"), secret_headers());

        let fixture = ScenarioRunner::record_live(&scenario, target).await.unwrap();

        assert_eq!(
            request_count.load(Ordering::SeqCst),
            2,
            "both turns should reach the backend"
        );
        assert_eq!(fixture.turns.len(), 2, "fixture should contain both turns");
        let first_client_response = &fixture.turns[0].client.response;
        let first_response_id = first_client_response.response_id();
        assert!(
            first_response_id.is_some(),
            "first Responses turn should produce a resp_-prefixed ID"
        );
        let RecordedBody::Json {
            value: second_request_body,
        } = &fixture.turns[1].client.request.body
        else {
            panic!("second turn client request should be JSON");
        };
        assert_eq!(
            second_request_body["previous_response_id"],
            first_response_id.unwrap(),
            "second turn must bind ${{PREVIOUS_RESPONSE_ID}} to the first turn's response ID"
        );
        provider.finish().await;
    }

    #[tokio::test]
    async fn record_live_rejects_configured_credential_in_scenario_request() {
        let secret = "scenario-request-credential";
        let provider = LocalProvider::start(handler(move |_request| async move { chat_response(0) })).await;
        let mut scenario = live_scenario();
        scenario.turns.truncate(1);
        let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
            panic!("live scenario request should be JSON");
        };
        value["messages"][0]["content"] = Value::String(secret.to_owned());

        let result = ScenarioRunner::record_live(
            &scenario,
            target(
                provider.http_url("/"),
                credential_headers(HeaderName::from_static("x-api-key"), secret),
            ),
        )
        .await;
        let Err(error) = result else {
            panic!("a configured credential in a request must not become a fixture");
        };

        assert_eq!(
            error.to_string(),
            "fixture commit safety violation: configured credential at $/turns/0/client/request/body/value/<key>/0/<key>"
        );
        assert_value_absent_from_error(&error, secret);
        provider.finish().await;
    }

    #[tokio::test]
    async fn record_live_rejects_json_escaped_configured_credential_without_exposing_it() {
        let secret = r#"json-\"credential\"\\tail"#;
        let provider = LocalProvider::start(handler(move |_request| async move {
            json_response(
                StatusCode::OK,
                &json!({
                    "id": "chatcmpl-reflected",
                    "model": "fixture-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": secret},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}
                }),
            )
        }))
        .await;
        let mut scenario = live_scenario();
        scenario.turns.truncate(1);
        let result = ScenarioRunner::record_live(
            &scenario,
            target(
                provider.http_url("/"),
                credential_headers(header::AUTHORIZATION, &format!("Bearer {secret}")),
            ),
        )
        .await;
        let Err(error) = result else {
            panic!("a JSON-escaped credential reflection must not become a fixture");
        };

        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_value_absent_from_error(&error, secret);
        provider.finish().await;
    }

    #[tokio::test]
    async fn recorder_rejects_safe_response_header_reflecting_configured_api_key() {
        let secret = "generic-header-credential";
        let provider = LocalProvider::start(handler(move |_request| async move {
            let mut response = json_response(StatusCode::OK, &json!({"ok": true}));
            response
                .headers_mut()
                .insert("request-id", HeaderValue::from_static(secret));
            response
        }))
        .await;
        let recorder = RecordingProxy::start(target(
            provider.http_url("/"),
            credential_headers(HeaderName::from_static("x-api-key"), secret),
        ))
        .await
        .unwrap();
        let response = authorized_client(&recorder)
            .post(format!("http://{}/v1/chat/completions", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model"}"#)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(!response.text().await.unwrap().contains(secret));
        let result = recorder.finish(1);
        let Err(error) = result else {
            panic!("a safe response header must not reflect a configured API key");
        };
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_value_absent_from_error(&error, secret);
        provider.finish().await;
    }

    #[tokio::test]
    async fn recorder_rejects_json_reflecting_a_custom_outbound_header_value() {
        let secret = "custom-outbound-credential";
        let provider = LocalProvider::start(handler(move |_request| async move {
            json_response(StatusCode::OK, &json!({"echo": secret}))
        }))
        .await;
        let recorder = RecordingProxy::start(target(
            provider.http_url("/"),
            credential_headers(HeaderName::from_static("x-auth-token"), secret),
        ))
        .await
        .unwrap();
        let response = authorized_client(&recorder)
            .post(format!("http://{}/v1/chat/completions", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model"}"#)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.bytes().await.unwrap();
        let error = recorder.finish(1).unwrap_err();
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_value_absent_from_error(&error, secret);
        provider.finish().await;
    }

    #[tokio::test]
    async fn recorder_rejects_sse_credential_split_across_provider_chunks() {
        let secret = "streaming-credential-sentinel";
        let provider = LocalProvider::start(handler(move |_request| async move {
            let chunks = [
                Bytes::from_static(b"data: {\"echo\":\"streaming-cre"),
                Bytes::from_static(b"dential-sentinel\"}\n\ndata: [DONE]\n\n"),
            ]
            .into_iter()
            .map(|chunk| Ok::<_, io::Error>(Frame::data(chunk)));
            let mut response = Response::new(http_body_util::BodyExt::boxed(StreamBody::new(stream::iter(chunks))));
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            response
        }))
        .await;
        let recorder = RecordingProxy::start(target(
            provider.http_url("/"),
            credential_headers(header::AUTHORIZATION, &format!("Bearer {secret}")),
        ))
        .await
        .unwrap();
        let response = authorized_client(&recorder)
            .post(format!("http://{}/v1/chat/completions", recorder.addr()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"model":"fixture-model","stream":true}"#)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let _body = response.bytes().await.unwrap();
        let result = recorder.finish(1);
        let Err(error) = result else {
            panic!("a cross-chunk SSE credential reflection must not complete");
        };
        assert_eq!(error.to_string(), "recording provider response capture was incomplete");
        assert_value_absent_from_error(&error, secret);
        provider.finish().await;
    }

    #[tokio::test]
    async fn record_live_refuses_transport_failure_instead_of_materializing_synthetic_502() {
        let provider = DisconnectProvider::start_before_headers().await;
        let mut scenario = live_scenario();
        scenario.turns.truncate(1);
        let error = ScenarioRunner::record_live(&scenario, target(provider.url(), secret_headers()))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "recording provider response failed before start");
        assert_secret_absent_from_error(&error);
        provider.finish().await;
    }

    #[tokio::test]
    async fn record_live_rejects_vacuous_turns_before_starting_network_resources() {
        let mut scenario = live_scenario();
        scenario.turns.clear();

        let error = ScenarioRunner::record_live(
            &scenario,
            target(UrlCase::parse("http://127.0.0.1:9/"), secret_headers()),
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            FixtureError::InvalidInferenceTurnCount {
                document: "inference scenario",
                count: 0
            }
        ));
        assert_secret_absent_from_error(&error);
    }

    #[tokio::test]
    async fn record_live_rejects_non_string_model_before_starting_network_resources() {
        let mut scenario = live_scenario();
        let RecordedBody::Json { value } = &mut scenario.turns[0].request.body else {
            panic!("live scenario request should be JSON");
        };
        value["model"] = json!(42);

        let error = ScenarioRunner::record_live(
            &scenario,
            target(UrlCase::parse("http://127.0.0.1:9/"), secret_headers()),
        )
        .await
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "recording scenario model does not match provider target"
        );
        assert_secret_absent_from_error(&error);
    }

    // Test Utilities
    // -------------------------------------------------------------------------

    struct StreamState {
        chunks: std::collections::VecDeque<Bytes>,
        index: usize,
        provider_waiting: Option<oneshot::Sender<()>>,
        release: Option<oneshot::Receiver<()>>,
    }

    struct LocalProvider {
        addr: SocketAddr,
        shutdown: Option<oneshot::Sender<()>>,
        task: JoinHandle<()>,
    }

    impl LocalProvider {
        async fn start(handler: Handler) -> Self {
            let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
                .await
                .unwrap();
            let addr = listener.local_addr().unwrap();
            let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
            let task = tokio::spawn(async move {
                let mut connections = JoinSet::new();
                loop {
                    tokio::select! {
                        _ = &mut shutdown_rx => break,
                        accepted = listener.accept() => {
                            let (stream, _) = accepted.unwrap();
                            let handler = Arc::clone(&handler);
                            connections.spawn(async move {
                                let service = service_fn(move |request| {
                                    let handler = Arc::clone(&handler);
                                    async move { Ok::<_, Infallible>(handler(request).await) }
                                });
                                let _result = http1::Builder::new()
                                    .keep_alive(false)
                                    .serve_connection(TokioIo::new(stream), service)
                                    .await;
                            });
                        },
                        _joined = connections.join_next(), if !connections.is_empty() => {},
                    }
                }
                drop(listener);
                while connections.join_next().await.is_some() {}
            });
            Self {
                addr,
                shutdown: Some(shutdown_tx),
                task,
            }
        }

        async fn finish(mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _sent = shutdown.send(());
            }
            tokio::time::timeout(TEST_TIMEOUT, &mut self.task)
                .await
                .expect("provider should shut down")
                .unwrap();
            assert_listener_released(self.addr).await;
        }

        fn http_url(&self, path: &str) -> reqwest::Url {
            reqwest::Url::parse(&format!("http://{}{path}", self.addr)).unwrap()
        }
    }

    struct TlsProvider {
        addr: SocketAddr,
        shutdown: Option<oneshot::Sender<()>>,
        task: JoinHandle<()>,
    }

    impl TlsProvider {
        async fn start(certificates: &TestCertificates) -> Self {
            let certificate_pem = std::fs::read(&certificates.cert_path).unwrap();
            let key_pem = std::fs::read(&certificates.key_path).unwrap();
            let certificate_chain = rustls_pemfile::certs(&mut &*certificate_pem)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let key = rustls_pemfile::private_key(&mut &*key_pem).unwrap().unwrap();
            let config =
                rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::aws_lc_rs::default_provider()))
                    .with_protocol_versions(rustls::DEFAULT_VERSIONS)
                    .unwrap()
                    .with_no_client_auth()
                    .with_single_cert(certificate_chain, key)
                    .unwrap();
            let acceptor = TlsAcceptor::from(Arc::new(config));
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
            let task = tokio::spawn(async move {
                tokio::select! {
                    _ = &mut shutdown_rx => {},
                    accepted = listener.accept() => {
                        let (stream, _) = accepted.unwrap();
                        let tls = acceptor.accept(stream).await.unwrap();
                        let service = service_fn(|_request| async {
                            Ok::<_, Infallible>(json_response(StatusCode::OK, &json!({"tls": "rustls"})))
                        });
                        http1::Builder::new()
                            .serve_connection(TokioIo::new(tls), service)
                            .await
                            .unwrap();
                    },
                }
                drop(listener);
            });
            Self {
                addr,
                shutdown: Some(shutdown_tx),
                task,
            }
        }

        async fn finish(mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _sent = shutdown.send(());
            }
            tokio::time::timeout(TEST_TIMEOUT, &mut self.task)
                .await
                .expect("TLS provider should shut down")
                .unwrap();
            assert_listener_released(self.addr).await;
        }

        fn url(&self, path: &str) -> reqwest::Url {
            reqwest::Url::parse(&format!("https://localhost:{}{path}", self.addr.port())).unwrap()
        }
    }

    struct DisconnectProvider {
        addr: SocketAddr,
        task: JoinHandle<()>,
    }

    impl DisconnectProvider {
        async fn start_before_headers() -> Self {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let addr = listener.local_addr().unwrap();
            let task = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0_u8; 1024];
                let _read = stream.read(&mut buffer).await;
                drop(stream);
                drop(listener);
            });
            Self { addr, task }
        }

        async fn finish(mut self) {
            tokio::time::timeout(TEST_TIMEOUT, &mut self.task)
                .await
                .expect("disconnect provider should stop")
                .unwrap();
        }

        fn url(&self) -> reqwest::Url {
            reqwest::Url::parse(&format!("http://{}/", self.addr)).unwrap()
        }
    }

    struct UrlCase;

    impl UrlCase {
        fn parse(value: &str) -> reqwest::Url {
            reqwest::Url::parse(value).unwrap()
        }
    }

    fn handler<F, Fut>(handler: F) -> Handler
    where
        F: Fn(Request<Incoming>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response<ProviderBody>> + Send + 'static,
    {
        Arc::new(move |request| Box::pin(handler(request)))
    }

    fn target(base_url: reqwest::Url, outbound_headers: HeaderMap) -> ProviderTarget {
        ProviderTarget {
            provider: "local-provider".to_owned(),
            model: "fixture-model".to_owned(),
            base_url,
            outbound_headers,
        }
    }

    fn secret_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {SECRET_SENTINEL}")).unwrap(),
        );
        headers
    }

    fn credential_headers(name: HeaderName, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        let mut value = HeaderValue::from_str(value).unwrap();
        value.set_sensitive(true);
        headers.insert(name, value);
        headers
    }

    fn test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    fn authorized_client(recorder: &super::RecordingProxyGuard) -> reqwest::Client {
        let mut headers = HeaderMap::new();
        headers.insert(RECORDER_CAPABILITY_HEADER.clone(), recorder.capability().clone());
        reqwest::Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    fn isolated_recorder_shared() -> RecorderShared {
        RecorderShared {
            client: test_client(),
            hop_client: test_client(),
            state: StdMutex::new(RecorderState {
                attempts: Vec::new(),
                excess_attempt: false,
                retained: CaptureUsage::default(),
            }),
            target: target(UrlCase::parse("http://127.0.0.1/"), HeaderMap::new()),
            capability: HeaderValue::from_static("test-recorder-capability"),
            max_attempts: MAX_INFERENCE_TURNS,
            retained_limit: LIVE_CAPTURE_BUDGET,
            terminal: Notify::new(),
        }
    }

    fn json_response(status: StatusCode, value: &Value) -> Response<ProviderBody> {
        let bytes = Bytes::from(serde_json::to_vec(value).unwrap());
        let mut response = Response::new(infallible_body(bytes));
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        response
    }

    fn chat_response(index: usize) -> Response<ProviderBody> {
        json_response(
            StatusCode::OK,
            &json!({
                "id": format!("chatcmpl-live-{index}"),
                "model": "fixture-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": format!("answer-{index}")},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}
            }),
        )
    }

    fn empty_response(status: StatusCode) -> Response<ProviderBody> {
        let mut response = Response::new(infallible_body(Bytes::new()));
        *response.status_mut() = status;
        response
    }

    fn infallible_body(bytes: Bytes) -> ProviderBody {
        Full::new(bytes).map_err(|never| match never {}).boxed()
    }

    fn with_recorder_capability(recorder: &super::RecordingProxyGuard, request: &str) -> String {
        let (head, body) = request.split_once("\r\n\r\n").unwrap();
        format!(
            "{head}\r\n{}: {}\r\n\r\n{body}",
            RECORDER_CAPABILITY_HEADER,
            recorder.capability().to_str().unwrap()
        )
    }

    async fn authorized_raw_request(recorder: &super::RecordingProxyGuard, request: &str) -> String {
        let request = with_recorder_capability(recorder, request);
        let mut stream = TcpStream::connect(recorder.addr()).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    fn live_scenario() -> InferenceScenario {
        InferenceScenario {
            version: 1,
            id: "messages/live-record".to_owned(),
            description: "two-turn live recording".to_owned(),
            protocol: InferenceProtocol::AnthropicMessages,
            example_config: "anthropic/messages-to-openai.yaml".to_owned(),
            upstream_authority: "127.0.0.1:8000".to_owned(),
            features: vec!["messages.request.minimal".to_owned()],
            turns: vec![live_turn("first", "first prompt"), live_turn("second", "second prompt")],
        }
    }

    fn live_turn(name: &str, prompt: &str) -> ScenarioTurn {
        ScenarioTurn {
            name: name.to_owned(),
            request: RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/messages".to_owned(),
                headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                body: RecordedBody::Json {
                    value: json!({
                        "model": "${MODEL}",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": prompt}],
                    }),
                },
            },
            expect: ScenarioExpectation {
                client_status: 200,
                client_body_kind: BodyKind::Json,
                upstream_path: "/v1/chat/completions".to_owned(),
                upstream_body_kind: BodyKind::Json,
                client_sse_events: Vec::new(),
                client_sse_repeatable_events: Vec::new(),
                client_sse_interleaved_events: Vec::new(),
                upstream_sse_events: Vec::new(),
                upstream_sse_repeatable_events: Vec::new(),
                upstream_sse_interleaved_events: Vec::new(),
            },
        }
    }

    fn live_responses_scenario() -> InferenceScenario {
        InferenceScenario {
            version: 1,
            id: "responses/live-record".to_owned(),
            description: "two-turn Responses live recording with continuation".to_owned(),
            protocol: InferenceProtocol::OpenaiResponses,
            example_config: "openai/responses/responses-to-chat-completions.yaml".to_owned(),
            upstream_authority: "127.0.0.1:3001".to_owned(),
            features: vec!["responses.chat.continuation".to_owned()],
            turns: vec![
                live_responses_turn("initial", "first prompt", None),
                live_responses_turn("continuation", "second prompt", Some("${PREVIOUS_RESPONSE_ID}")),
            ],
        }
    }

    fn live_responses_turn(name: &str, prompt: &str, previous_response_id: Option<&str>) -> ScenarioTurn {
        let mut value = json!({
            "model": "${MODEL}",
            "input": prompt,
            "store": true,
            "stream": false,
        });
        if let Some(prev_id) = previous_response_id {
            value["previous_response_id"] = json!(prev_id);
        }
        ScenarioTurn {
            name: name.to_owned(),
            request: RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/responses".to_owned(),
                headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                body: RecordedBody::Json { value },
            },
            expect: ScenarioExpectation {
                client_status: 200,
                client_body_kind: BodyKind::Json,
                upstream_path: "/v1/chat/completions".to_owned(),
                upstream_body_kind: BodyKind::Json,
                client_sse_events: Vec::new(),
                client_sse_repeatable_events: Vec::new(),
                client_sse_interleaved_events: Vec::new(),
                upstream_sse_events: Vec::new(),
                upstream_sse_repeatable_events: Vec::new(),
                upstream_sse_interleaved_events: Vec::new(),
            },
        }
    }

    fn assert_secret_absent_from_error(error: &FixtureError) {
        assert_value_absent_from_error(error, SECRET_SENTINEL);
    }

    fn assert_value_absent_from_error(error: &FixtureError, value: &str) {
        let mut rendered = format!("{error}\n{error:?}");
        let mut source = error.source();
        while let Some(error) = source {
            write!(rendered, "\n{error}\n{error:?}").unwrap();
            source = error.source();
        }
        assert!(!rendered.contains(value));
    }
}
