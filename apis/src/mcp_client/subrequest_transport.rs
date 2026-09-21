// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Route the outbound MCP Streamable-HTTP exchange through a filtered
//! subrequest.
//!
//! This is the `#957` seam: instead of dialing the MCP server with an inline
//! `reqwest` client, the [`rmcp`] [`StreamableHttpClient`] adapter here targets
//! the dynamic MCP URL through praxis's URL-aware target preparation
//! ([`prepare_url_target`]) and runs the request through a bound outbound
//! [`FilteredSubrequestExecutor`] chain. The chain reuses praxis's own SSRF/DNS
//! validation, TLS/SNI handling, Host binding, deadline, and response-size
//! guardrails rather than reimplementing them per call.
//!
//! Both POST→SSE (client-initiated request/response that may stream) and
//! server-initiated GET SSE streams are supported through the filtered subrequest
//! path. Buffered exchanges (JSON responses) are parsed directly, and streaming
//! `text/event-stream` responses are forwarded incrementally via the SSE byte
//! adapter.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use bytes::Bytes;
use futures::stream::BoxStream;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
#[cfg(test)]
use praxis_core::subrequest::SubRequestConnector;
use praxis_core::{
    config::ChainRef,
    connectivity::{UrlTargetError, prepare_url_target},
    subrequest::{DEPTH_HEADER, SubRequestClient},
};
use praxis_filter::{
    BodyMode, CalloutOutcome, CalloutResponse, ChainBindingContext, FilterEntry, FilterError, FilterPipeline,
    FilterRegistry, FilteredSubrequestExecutor, HttpFilterContext, IterationState, RequestExtensions, StagedUpstream,
    StagedUpstreamFallback, StreamingResponseBody, SubRequest, SubResponse, SubrequestRuntime,
};
use rmcp::{
    model::{ClientJsonRpcMessage, ClientRequest, JsonRpcMessage, ServerJsonRpcMessage},
    transport::{
        common::http_header::{HEADER_LAST_EVENT_ID, HEADER_SESSION_ID},
        streamable_http_client::{
            AuthRequiredError, InsufficientScopeError, StreamableHttpClient, StreamableHttpError,
            StreamableHttpPostResponse,
        },
    },
};
use sse_stream::{Error as SseError, Sse};

use super::{McpClientError, McpDisplayUrl};

/// Wire byte ceiling for a control-plane MCP response.
///
/// Bounds the buffered body of `initialize`, each `tools/list` page, and any
/// other non-`tools/call` exchange before deserialization, so an untrusted
/// server cannot exhaust proxy memory with an oversized control response. Ported
/// from the previous reqwest-layer `bounded_http` client; this is now the single
/// home for the MCP response tiers, since the dial runs through the executor.
/// `mod.rs` derives its cumulative `tools/list` budget from this value.
pub(super) const MAX_CONTROL_RESPONSE_BYTES: usize = 1_048_576;

/// JSON-RPC envelope allowance added on top of the configured `tools/call`
/// result cap, covering the surrounding result object beyond the raw payload.
const MAX_TOOL_RESULT_ENVELOPE_BYTES: usize = 65_536;

/// Worst-case expansion factor for a UTF-8 payload rendered as JSON (`\u00XX`
/// escaping), applied to the configured `tools/call` result cap so the wire
/// ceiling admits any result that fits within the decoded cap.
const MAX_JSON_STRING_EXPANSION: usize = 6;

/// Default per-event SSE payload cap used when rmcp does not supply an explicit maximum.
const DEFAULT_MAX_SSE_EVENT_SIZE: usize = 16 * 1024 * 1024;

/// Translate a configured *decoded* `tools/call` result cap into the *wire* byte
/// ceiling to enforce before deserialization (worst-case JSON expansion plus the
/// JSON-RPC envelope allowance).
fn tool_result_wire_cap(max_result_bytes: usize) -> usize {
    max_result_bytes
        .saturating_mul(MAX_JSON_STRING_EXPANSION)
        .saturating_add(MAX_TOOL_RESULT_ENVELOPE_BYTES)
}

/// Loose executor backstop for a streaming callout (spec §4.5, F3).
///
/// praxis always arms its outer `CalloutStreamingBody` at the `max_response_bytes`
/// passed to [`McpSubrequestClient::execute_streaming`] and rejects an overflowing
/// chunk with an opaque `FilterError`, discarding our typed 413. If that ceiling
/// equalled the adapter's binding cap, praxis would trip *first* and the
/// transport's own `try_unfold` counter would never record
/// [`TransportSignal::ResponseTooLarge`]. So the executor ceiling is loosened to
/// twice the binding adapter cap: the adapter counter (still keyed to the exact
/// per-message/cumulative cap) trips first for any single chunk up to the full cap
/// size and yields HTTP 413, while the ceiling stays finite (not `usize::MAX`) so
/// the `None`-body buffered fallback and the success-`application/json` drain
/// (`collect_body`/`drain_body`) remain memory-bounded — at 2x the cap.
fn streaming_executor_backstop(binding_cap: usize) -> usize {
    binding_cap.saturating_mul(2)
}

/// The `mcp-session-id` header carrying the Streamable-HTTP session token.
fn session_id_header() -> HeaderName {
    HeaderName::from_static("mcp-session-id")
}

/// Request-extension marker that arms the streaming subrequest path.
///
/// Inserted into the callout's [`RequestExtensions`] only by the streaming
/// entry points ([`McpSubrequestClient::execute_streaming`] and the GET-stream
/// path). The auto-injected [`McpStreamingSelectorFilter`](crate::openai::McpStreamingSelectorFilter)
/// reads it during `on_request` and, when present, sets the subrequest response
/// mode to streaming. Its absence keeps every other callout (buffered
/// `initialize`/`tools/list`/`tools/call`, DELETE) on the unchanged buffered ladder.
#[derive(Debug, Clone, Copy)]
pub(crate) struct McpStreamingRequested;

// -----------------------------------------------------------------------------
// McpTransportError
// -----------------------------------------------------------------------------

/// Failure categories for the filtered-subrequest MCP transport.
///
/// Every variant is credential-safe: its `Display` never echoes the MCP URL,
/// userinfo, query, or any header value. This is the `Error` associated type of
/// [`McpSubrequestClient`], surfaced to `rmcp` as
/// [`StreamableHttpError::Client`].
#[derive(Debug, thiserror::Error)]
pub(crate) enum McpTransportError {
    /// The filtered subrequest failed to complete the exchange (transport
    /// failure, deadline, or an unexpected streaming response).
    #[error("mcp subrequest transport failed")]
    Transport,

    /// The MCP target URL could not be prepared into a dial target.
    #[error("mcp target preparation failed")]
    Target,

    /// The MCP target resolved to an address the SSRF policy rejected.
    ///
    /// [`McpSubrequestClient::execute`] records the typed classification
    /// out-of-band (see [`TransportSignal`]) before returning this variant, so
    /// the free-standing caller can reconstruct the SSRF rejection after `rmcp`
    /// discards the typed error.
    #[error("mcp target blocked by address policy")]
    SsrfBlocked,

    /// The outbound JSON-RPC message could not be serialized.
    #[error("failed to serialize mcp request")]
    Serialize,

    /// The server returned a status or a session/auth header the transport
    /// could not represent.
    #[error("mcp server returned an unrepresentable status or header")]
    InvalidStatus,

    /// The outbound MCP pipeline or TLS material could not be constructed.
    #[error("failed to set up the mcp outbound path")]
    Setup,

    /// The server's response exceeded the configured size limit.
    ///
    /// The filtered callout classifies the oversized response as
    /// [`CalloutOutcome::ResponseTooLarge`]; the transport records the typed
    /// overflow out-of-band (see [`TransportSignal`]) and returns this variant so
    /// the caller can map it to HTTP 413.
    #[error("mcp server returned a response exceeding the size limit")]
    ResponseTooLarge,
}

// -----------------------------------------------------------------------------
// TransportSignal
// -----------------------------------------------------------------------------

/// A typed transport classification observed on an MCP callout, carried
/// out-of-band past `rmcp`'s opaque transport-error mapping.
///
/// `rmcp`'s `StreamableHttpClient` funnels every transport failure through an
/// opaque `.map_err(|_source| …)` that discards the typed [`McpTransportError`],
/// so neither a [`CalloutOutcome::ResponseTooLarge`] classification nor an SSRF
/// address rejection can ride out on the returned `StreamableHttpError`.
/// [`McpSubrequestClient::execute`] records the classification here instead
/// (first signal wins), and the free-standing caller in `mod.rs` reads it back
/// via [`transport_signal_error`] after the rmcp `serve`/pagination call fails.
#[derive(Clone, Copy)]
pub(crate) enum TransportSignal {
    /// The server's response exceeded the effective size limit; maps to HTTP 413.
    ResponseTooLarge {
        /// The effective response-size limit that was exceeded.
        limit: usize,
    },
    /// The MCP target resolved to an address the SSRF policy rejected.
    SsrfBlocked,
    /// The MCP target URL was structurally invalid or disallowed at parse time
    /// (bad scheme, embedded userinfo, a fragment, or a malformed/disallowed host
    /// literal) — rejected before any dial. Distinct from [`Self::SsrfBlocked`]
    /// only in the surfaced error text; both are permanent, hard rejections.
    TargetRejected,
}

// -----------------------------------------------------------------------------
// Outbound chain construction
// -----------------------------------------------------------------------------

/// Build the auto-injected selector chain entry by deserializing through
/// [`FilterEntry`]'s own path (the exact path operator configs use), so the
/// entry is always structurally valid and free of conditions.
#[expect(clippy::expect_used, reason = "static streaming-selector entry YAML is always valid")]
fn streaming_selector_entry() -> FilterEntry {
    serde_yaml::from_str(&format!(
        "filter: {name}",
        name = super::McpStreamingSelectorFilter::NAME
    ))
    .expect("static streaming-selector entry YAML is always valid")
}

/// Return the outbound chain with the streaming selector guaranteed first.
///
/// `None` and inline chains are rewritten with the selector prepended. A named
/// chain is returned untouched (we cannot mutate a shared named chain); its
/// structural requirement — the selector must be its first, unconditional
/// filter — is validated after binding in [`bind_mcp_outbound_chain`].
fn selector_injected_chain(outbound_chain: Option<ChainRef>, chain_name: &str) -> ChainRef {
    match outbound_chain {
        None => ChainRef::Inline {
            name: chain_name.to_owned(),
            filters: vec![streaming_selector_entry()],
        },
        Some(ChainRef::Inline { name, mut filters }) => {
            filters.insert(0, streaming_selector_entry());
            ChainRef::Inline { name, filters }
        },
        Some(named @ ChainRef::Named(_)) => named,
    }
}

/// Resolve a chain-binding MCP filter's configured `outbound_chain` into a bound
/// outbound [`FilterPipeline`].
///
/// No upstream-selecting filter is needed: [`McpSubrequestClient::execute`]
/// stages a [`StagedUpstream`] (and a [`StagedUpstreamFallback`]) in the
/// per-request extensions, and the executor seeds the nested context's upstream
/// from it before the request phase. The bound chain therefore carries only the
/// operator's cross-cutting filters (observability, security, credentials),
/// which observe and may act on the outbound MCP request while the callout keeps
/// full control of the validated destination.
///
/// Both reference shapes are honored, because this runs at top-level build time
/// with a live [`ChainBindingContext`]: an inline chain is bound directly, and a
/// [`ChainRef::Named`] reference resolves against the top-level `filter_chains`.
/// When `outbound_chain` is [`None`] the bound chain is an empty inline chain
/// named `chain_name`, so the callout dials the staged upstream with no extra
/// filters.
///
/// # Errors
///
/// Returns [`FilterError`] if [`ChainBindingContext::bind_chain`] rejects the
/// chain (an unknown named reference, a cycle, excessive nesting, a terminal
/// filter, or an ordering violation).
pub(crate) fn bind_mcp_outbound_chain(
    outbound_chain: Option<ChainRef>,
    ctx: &ChainBindingContext<'_>,
    chain_name: &str,
) -> Result<Arc<FilterPipeline>, FilterError> {
    let is_named = matches!(outbound_chain, Some(ChainRef::Named(_)));
    let chain_ref = selector_injected_chain(outbound_chain, chain_name);
    let pipeline = ctx.bind_chain(&chain_ref)?;

    // A named chain is bound verbatim; require the selector to be its first,
    // unconditional filter. Match on the resolved filter *type* name (not a
    // user-assigned label) and reject conditions that could skip it.
    if is_named {
        let snapshot = pipeline.introspection();
        let ok = snapshot
            .first()
            .is_some_and(|f| f.filter == super::McpStreamingSelectorFilter::NAME && f.conditions.is_empty());
        if !ok {
            return Err(FilterError::from(format!(
                "openai_mcp_streaming_selector: named outbound_chain '{chain_name}' must list \
                 '{name}' as its first, unconditional filter to support SSE streaming",
                name = super::McpStreamingSelectorFilter::NAME
            )));
        }
    }

    // A response-body-buffering filter anywhere in the chain defeats streaming.
    // Reject at bind time; this is immune to `skip_pipeline_validation`.
    if matches!(
        pipeline.body_capabilities().response_body_mode,
        BodyMode::StreamBuffer { .. }
    ) {
        return Err(FilterError::from(format!(
            "openai_mcp_streaming_selector: outbound_chain '{chain_name}' contains a filter that \
             buffers the response body (StreamBuffer), which is incompatible with SSE streaming"
        )));
    }

    Ok(Arc::new(pipeline))
}

/// Build the bare, empty outbound pipeline used when no operator `outbound_chain`
/// is bound.
///
/// No upstream-selecting filter is needed: [`McpSubrequestClient::execute`]
/// stages a [`StagedUpstream`] the executor seeds the dial target from before the
/// request phase, so a zero-filter pipeline dials the validated destination
/// directly. `allow_private` mirrors the operator's global private-upstream
/// posture: the executor's peer builder consults
/// [`FilterPipeline::allow_private_upstreams`] and refuses private/reserved dial
/// targets unless it is set, so the pipeline must carry the same policy the
/// [`ssrf_validate`] hook enforces. Plain-builtin filter registration
/// (`from_config`) builds this with the safe default posture (private denied) and
/// lets pipeline finalization propagate the operator's global insecure options.
/// Unit tests that construct a callout without a live pipeline build
/// (`McpCallout::fabricated`) bake the posture directly.
///
/// # Errors
///
/// Returns [`McpClientError::Connection`] if the empty pipeline cannot be built.
pub(crate) fn build_bare_outbound_pipeline(allow_private: bool) -> Result<Arc<FilterPipeline>, McpClientError> {
    let registry = FilterRegistry::with_builtins();
    let mut entries: Vec<FilterEntry> = Vec::new();
    let mut pipeline = FilterPipeline::build(&mut entries, &registry).map_err(|_error| McpClientError::Connection {
        url: McpDisplayUrl::invalid(),
    })?;
    pipeline.set_allow_private_upstreams(allow_private);
    Ok(Arc::new(pipeline))
}

// -----------------------------------------------------------------------------
// McpCallout
// -----------------------------------------------------------------------------

/// The request-scoped resources an MCP callout dials through.
///
/// This bundles the *real* parent subrequest transport and downstream
/// attributes read off the owning [`HttpFilterContext`] together with the bound
/// outbound [`FilterPipeline`] the chain-binding MCP filter resolved at build
/// time. Both `openai_mcp_tool_resolve` and `openai_mcp_dispatch` construct one
/// of these per request via [`from_context`](Self::from_context) and thread it
/// down to
/// [`list_tools_with_forwarded_headers`](super::list_tools_with_forwarded_headers)/
/// [`call_tool_with_forwarded_headers`](super::call_tool_with_forwarded_headers), which build an
/// [`McpSubrequestClient`] from it.
///
/// It deliberately does *not* fabricate a fresh connector or a default runtime:
/// the callout must share the server's connection pool and carry the originating
/// client's address, TLS posture, peer identity, and request start so the nested
/// pipeline sees the same downstream context the top-level request does.
#[derive(Clone)]
pub(crate) struct McpCallout {
    /// Shared parent sub-request transport (connection pool included).
    client: SubRequestClient,
    /// Downstream attributes captured from the owning request context.
    downstream: SubrequestRuntime,
    /// Operator-configured (or empty) outbound pipeline for the callout; the
    /// dial target is staged separately via [`StagedUpstream`], not selected by
    /// a pipeline filter.
    pipeline: Arc<FilterPipeline>,
    /// Sub-request nesting depth for callouts issued from this context.
    depth: u8,
    /// Whether private/loopback MCP destinations are permitted, taken from the
    /// bound pipeline's finalized posture (never a per-filter opt-in).
    allow_private: bool,
}

impl McpCallout {
    /// Capture the parent transport and downstream attributes from `ctx`,
    /// pairing them with the already-bound outbound `pipeline`.
    ///
    /// Returns [`None`] when the pipeline exposes no shared sub-request client
    /// (for example a pipeline built without a server runtime): an MCP callout
    /// cannot proceed without the parent transport, so the caller fails closed.
    ///
    /// `allow_private` is read from the bound pipeline
    /// ([`FilterPipeline::allow_private_upstreams`]), which pipeline
    /// finalization sets from the operator's global insecure posture — the same
    /// value the executor's peer builder enforces — so the SSRF hook here and the
    /// executor agree without a per-filter flag.
    pub(crate) fn from_context(ctx: &HttpFilterContext<'_>, pipeline: Arc<FilterPipeline>) -> Option<Self> {
        let client = ctx.subrequest_client?.clone();
        let downstream = SubrequestRuntime::new(
            ctx.client_addr,
            ctx.downstream_tls,
            ctx.peer_identity.clone(),
            ctx.request_start,
        );
        let allow_private = pipeline.allow_private_upstreams();
        let depth = resolve_callout_depth(
            ctx.extensions.get::<IterationState>().map(IterationState::depth),
            &ctx.request.headers,
        );
        Some(Self {
            client,
            downstream,
            pipeline,
            depth,
            allow_private,
        })
    }

    /// Whether private/loopback MCP destinations are permitted for this callout.
    pub(crate) fn allow_private(&self) -> bool {
        self.allow_private
    }

    /// Build a callout backed by a fabricated connector and a bare, empty
    /// outbound pipeline, for unit tests that exercise the transport without a
    /// live pipeline build.
    ///
    /// # Errors
    ///
    /// Returns [`McpClientError`] if the bare outbound pipeline cannot be built.
    #[cfg(test)]
    pub(crate) fn fabricated(allow_private: bool) -> Result<Self, McpClientError> {
        let pipeline = build_bare_outbound_pipeline(allow_private)?;
        Ok(Self {
            client: SubRequestClient::new(SubRequestConnector::new(1, None)),
            downstream: SubrequestRuntime::new(None, false, None, Instant::now()),
            pipeline,
            depth: 0,
            allow_private,
        })
    }
}

/// Resolve the sub-request nesting depth for an MCP callout issued from a
/// filter context.
///
/// The executor increments this value for the nested request
/// ([`FrameworkHeaders::set_depth`] emits `self.depth + 1`), so it must reflect
/// the depth of the *current* request, not a fixed zero. Sources, in order:
///
/// 1. The iterative-router's [`IterationState::depth`], when the callout is issued from inside an IRR step (the
///    dispatch filter's case).
/// 2. Otherwise the reserved `x-praxis-iterative-depth` header on the incoming request, set by the framework when this
///    proxy is itself reached as a nested sub-request. The header uses the `x-praxis-*` reserved prefix, so ingress
///    rejects client-spoofed values; a malformed value falls back to 0.
/// 3. Otherwise 0, for a genuine top-level request.
///
/// Hard-coding 0 would let a nested callout under-report its depth and defeat the
/// executor's loop-prevention bound.
///
/// [`FrameworkHeaders::set_depth`]: praxis_core::subrequest::FrameworkHeaders::set_depth
fn resolve_callout_depth(iteration_state_depth: Option<u8>, headers: &HeaderMap) -> u8 {
    if let Some(depth) = iteration_state_depth {
        return depth;
    }
    headers
        .get(DEPTH_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.parse::<u8>().ok())
        .unwrap_or(0)
}

// -----------------------------------------------------------------------------
// McpSubrequestClient
// -----------------------------------------------------------------------------

/// `rmcp` Streamable-HTTP client that routes each exchange through a filtered
/// subrequest.
#[derive(Clone)]
pub(crate) struct McpSubrequestClient {
    /// Request-scoped transport, downstream attributes, and bound pipeline.
    callout: McpCallout,
    /// Wire byte ceiling applied to a `tools/call` response on this client.
    ///
    /// Control-plane exchanges (`initialize`, `tools/list`, ...) are always
    /// bounded to [`MAX_CONTROL_RESPONSE_BYTES`]; only `tools/call` responses use
    /// this configured, JSON-expansion-adjusted ceiling (see [`Self::response_limit`]).
    tool_result_bytes: usize,
    /// Cumulative wire-byte ceiling for a server-initiated GET SSE stream.
    ///
    /// An intentionally coarse raw-wire `DoS` backstop, not decoded parity: for a
    /// control client it is `MAX_LISTING_RESPONSE_BYTES + MAX_CONTROL_RESPONSE_BYTES`
    /// (5 MiB); `paginate_tools` remains the authoritative decoded gate. Read
    /// only by the GET-stream path.
    stream_cumulative_cap: usize,
    /// Per-exchange duration ceiling.
    step_timeout: Duration,
    /// Out-of-band record of a typed classification observed on a callout.
    ///
    /// `rmcp` discards the typed [`McpTransportError`] on failure, so a
    /// [`CalloutOutcome::ResponseTooLarge`] classification or an SSRF address
    /// rejection is recorded here (first signal wins) and read back by the caller
    /// via [`transport_signal_error`] after the rmcp `serve`/pagination call
    /// fails. Shared through the [`Clone`] the transport requires, so the handle
    /// taken before the client is moved into the rmcp transport observes writes
    /// made during the exchange.
    signal: Arc<OnceLock<TransportSignal>>,
}

impl McpSubrequestClient {
    /// Build a client for the control-plane exchanges performed by
    /// [`list_tools_with_forwarded_headers`](super::list_tools_with_forwarded_headers):
    /// `initialize` and `tools/list`.
    ///
    /// No `tools/call` result flows over this transport, so every response is
    /// bounded to [`MAX_CONTROL_RESPONSE_BYTES`] before deserialization.
    pub(crate) fn control(callout: McpCallout, step_timeout: Duration) -> Self {
        Self::with_wire_cap(
            callout,
            step_timeout,
            MAX_CONTROL_RESPONSE_BYTES,
            crate::mcp_client::MAX_LISTING_RESPONSE_BYTES.saturating_add(MAX_CONTROL_RESPONSE_BYTES),
        )
    }

    /// Build a client for
    /// [`call_tool_with_forwarded_headers`](super::call_tool_with_forwarded_headers):
    /// `initialize` uses the control ceiling and the `tools/call` response is
    /// bounded to the configured `max_result_bytes` cap, expanded for worst-case
    /// JSON string escaping.
    ///
    /// `step_timeout` bounds each individual HTTP exchange; the `callout` carries
    /// the parent transport and the bound outbound pipeline whose finalized
    /// posture decides whether loopback destinations are permitted.
    pub(crate) fn for_tool(callout: McpCallout, step_timeout: Duration, max_result_bytes: usize) -> Self {
        let wire = tool_result_wire_cap(max_result_bytes);
        Self::with_wire_cap(
            callout,
            step_timeout,
            wire,
            wire.saturating_add(MAX_CONTROL_RESPONSE_BYTES),
        )
    }

    /// Shared constructor: move in the callout and pin the `tools/call` wire
    /// ceiling.
    fn with_wire_cap(
        callout: McpCallout,
        step_timeout: Duration,
        tool_result_bytes: usize,
        stream_cumulative_cap: usize,
    ) -> Self {
        Self {
            callout,
            tool_result_bytes,
            step_timeout,
            stream_cumulative_cap,
            signal: Arc::new(OnceLock::new()),
        }
    }

    /// Take a handle to this client's transport-signal record before the client
    /// is moved into the rmcp transport.
    ///
    /// The returned [`Arc`] aliases the same [`OnceLock`] the transport writes to
    /// during an exchange, so the caller can read back a
    /// [`CalloutOutcome::ResponseTooLarge`] classification or an SSRF address
    /// rejection after the rmcp `serve`/pagination call fails (see
    /// [`transport_signal_error`]).
    pub(crate) fn signal_handle(&self) -> Arc<OnceLock<TransportSignal>> {
        Arc::clone(&self.signal)
    }

    /// Per-event wire ceiling for a GET SSE stream (the tool-result wire cap).
    fn wire_cap(&self) -> usize {
        self.tool_result_bytes
    }

    /// Cumulative wire ceiling for a GET SSE stream.
    fn stream_cumulative_cap(&self) -> usize {
        self.stream_cumulative_cap
    }

    /// Select the wire byte ceiling for one outbound message.
    ///
    /// `tools/call` responses use the configured tool-result ceiling; every other
    /// exchange (`initialize`, `tools/list`, notifications, ...) is bounded to the
    /// control ceiling so an untrusted server cannot exhaust proxy memory on the
    /// control plane.
    fn response_limit(&self, message: &ClientJsonRpcMessage) -> usize {
        match message {
            ClientJsonRpcMessage::Request(request) if matches!(request.request, ClientRequest::CallToolRequest(_)) => {
                self.tool_result_bytes
            },
            _ => MAX_CONTROL_RESPONSE_BYTES,
        }
    }

    /// Prepare and validate the dial for `uri` — SSRF/DNS validation, upstream
    /// staging, and executor construction — returning the staged executor,
    /// request, extensions, and deadline for the caller to run.
    #[expect(
        clippy::too_many_lines,
        reason = "linear prepare/validate/dial sequence reads clearest inline"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "method/uri/body/headers/limit describe one dial call"
    )]
    async fn prepare_staged_request(
        &self,
        method: Method,
        uri: &str,
        body: Bytes,
        headers: HeaderMap,
        max_response_bytes: usize,
    ) -> Result<
        (FilteredSubrequestExecutor, SubRequest, RequestExtensions, Instant),
        StreamableHttpError<McpTransportError>,
    > {
        let deadline = Instant::now()
            .checked_add(self.step_timeout)
            .ok_or(StreamableHttpError::Client(McpTransportError::Setup))?;
        let allow_private = self.callout.allow_private;
        let target = match prepare_url_target(uri, deadline, move |addrs| ssrf_validate(addrs, allow_private)).await {
            Ok(target) => target,
            Err(error) => {
                if let Some(signal) = prepare_error_signal(&error) {
                    // First signal wins; the free-standing caller reconstructs the
                    // typed hard rejection (SSRF or invalid target) after rmcp
                    // discards the transport error (see `transport_signal_error`).
                    // Transient failures (DNS, deadline) record nothing and fall
                    // back to the caller's generic connection error.
                    self.signal.get_or_init(|| signal);
                }
                return Err(map_prepare_error(&error));
            },
        };
        let staged = StagedUpstream::from_prepared_target(&target)
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::Setup))?;
        let fallback = StagedUpstreamFallback::from_prepared_target(&target);
        let origin = origin_form(uri).map_err(StreamableHttpError::Client)?;

        let request = SubRequest {
            method,
            uri: origin,
            headers,
            body,
        };
        // Stage the validated dial target (and its DNS-failover set) so the
        // executor seeds the nested context's upstream before the request phase.
        // The bound outbound pipeline therefore never has to select the upstream.
        let mut extensions = RequestExtensions::default();
        extensions.insert(staged);
        extensions.insert(fallback);

        let executor = FilteredSubrequestExecutor::for_callout(
            self.callout.client.clone(),
            self.callout.downstream.clone(),
            self.callout.depth,
            max_response_bytes,
            self.step_timeout,
        );
        Ok((executor, request, extensions, deadline))
    }

    /// Prepare, validate, and dial `uri`, returning the buffered response.
    ///
    /// SSRF/DNS validation, TLS/SNI, Host binding, and the response-size ceiling
    /// are enforced by [`prepare_url_target`] and the executor.
    #[expect(clippy::large_stack_frames, reason = "rmcp/executor futures are inherently large")]
    #[expect(
        clippy::too_many_arguments,
        reason = "method/uri/body/headers/limit describe one dial call"
    )]
    async fn execute(
        &self,
        method: Method,
        uri: &str,
        body: Bytes,
        headers: HeaderMap,
        max_response_bytes: usize,
    ) -> Result<SubResponse, StreamableHttpError<McpTransportError>> {
        let (executor, request, extensions, deadline) = self
            .prepare_staged_request(method, uri, body, headers, max_response_bytes)
            .await?;
        let outcome = Box::pin(executor.run_classified(&self.callout.pipeline, &request, extensions, deadline))
            .await
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::Transport))?;
        match outcome {
            CalloutOutcome::Response(CalloutResponse::Buffered(response)) => Ok(response),
            CalloutOutcome::ResponseTooLarge { actual, limit } => {
                tracing::debug!(actual = ?actual, limit, "mcp callout response exceeded size limit");
                // First signal wins; the caller reads this back after rmcp
                // discards the typed error (see `transport_signal_error`).
                self.signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
            },
            // A single request/response MCP exchange never selects streaming, and
            // `CalloutOutcome` is `#[non_exhaustive]`: fail closed on any other
            // outcome (streaming or a future variant).
            _ => Err(StreamableHttpError::Client(McpTransportError::Transport)),
        }
    }

    /// Build the header map for a POST message, injecting Accept, Content-Type,
    /// and the optional session-id header.
    #[expect(
        clippy::unused_self,
        reason = "method form matches the refactored post_message call site"
    )]
    fn build_post_headers(
        &self,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        session_id: Option<&Arc<str>>,
    ) -> Result<HeaderMap, StreamableHttpError<McpTransportError>> {
        let mut headers = build_request_headers(auth_header, custom_headers)?;
        headers.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("text/event-stream, application/json"),
        );
        headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(session) = session_id {
            let value = HeaderValue::from_str(session)
                .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
            headers.insert(session_id_header(), value);
        }
        Ok(headers)
    }

    /// Build the header map for a GET stream, injecting Accept, session-id, and
    /// optional Last-Event-ID for resumption.
    #[expect(
        clippy::unused_self,
        reason = "method form matches the get_stream_with_max_sse_event_size call site"
    )]
    fn build_get_stream_headers(
        &self,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        session_id: Option<&Arc<str>>,
        last_event_id: Option<&str>,
    ) -> Result<HeaderMap, StreamableHttpError<McpTransportError>> {
        let mut headers = build_request_headers(auth_header, custom_headers)?;
        // GET SSE streams accept only text/event-stream.
        headers.insert(http::header::ACCEPT, HeaderValue::from_static("text/event-stream"));
        if let Some(session) = session_id {
            let value = HeaderValue::from_str(session)
                .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
            headers.insert(session_id_header(), value);
        }
        if let Some(id) = last_event_id {
            // Set from the rmcp-owned resumption cursor; a caller-supplied
            // Last-Event-ID was already rejected by build_request_headers.
            let value = HeaderValue::from_str(id)
                .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
            headers.insert(HEADER_LAST_EVENT_ID, value);
        }
        Ok(headers)
    }

    /// Classify a GET stream response into an rmcp SSE stream or a hard rejection.
    ///
    /// For the `200 text/event-stream` forward path, the body is moved into the
    /// SSE adapter and ownership transfers to the returned stream. For every other
    /// classification (405, 404, any non-SSE success), the body is cancelled
    /// before the error is returned.
    async fn classify_get_stream_response(
        &self,
        response: SubResponse,
        body: Option<Box<dyn StreamingResponseBody>>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<McpTransportError>> {
        let status = StatusCode::from_u16(response.status)
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
        let content_type = response
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok());

        let is_sse = status.is_success() && content_type.is_some_and(is_event_stream_content_type);
        if !is_sse {
            // 405 / 404 / any non-SSE success: this server has no GET stream.
            // best-effort cancel the forwarded body before rejecting (it is None
            // only when praxis buffered, handled by the Some guard below).
            if let Some(mut body) = body {
                body.cancel().await;
            }
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        }

        let Some(body) = body else {
            // Success + SSE content type but praxis buffered: no stream to forward.
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        };

        Ok(crate::mcp_client::sse_adapter::sse_stream_from_body(
            body,
            self.wire_cap(),
            self.stream_cumulative_cap(),
            max_sse_event_size,
            self.signal_handle(),
        ))
    }

    /// Streaming twin of [`Self::execute`]: arms the callout for streaming and returns
    /// the header response plus the streaming body when praxis selected streaming.
    ///
    /// Returns `(response, None)` when the callout was buffered anyway (Blocker 5:
    /// the full buffered body is preserved on `response.body`), and records a 413
    /// signal on [`CalloutOutcome::ResponseTooLarge`].
    #[expect(clippy::large_stack_frames, reason = "rmcp/executor futures are inherently large")]
    #[expect(
        clippy::too_many_arguments,
        reason = "method/uri/body/headers/limit describe one dial call"
    )]
    async fn execute_streaming(
        &self,
        method: Method,
        uri: &str,
        body: Bytes,
        headers: HeaderMap,
        max_response_bytes: usize,
    ) -> Result<(SubResponse, Option<Box<dyn StreamingResponseBody>>), StreamableHttpError<McpTransportError>> {
        let (executor, request, mut extensions, deadline) = self
            .prepare_staged_request(method, uri, body, headers, max_response_bytes)
            .await?;
        extensions.insert(McpStreamingRequested);
        let outcome = Box::pin(executor.run_classified(&self.callout.pipeline, &request, extensions, deadline))
            .await
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::Transport))?;
        match outcome {
            CalloutOutcome::Response(CalloutResponse::Streaming { response, body }) => Ok((response, Some(body))),
            CalloutOutcome::Response(CalloutResponse::Buffered(response)) => Ok((response, None)),
            CalloutOutcome::ResponseTooLarge { actual, limit } => {
                tracing::debug!(actual = ?actual, limit, "mcp streaming callout response exceeded size limit");
                self.signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
            },
            _ => Err(StreamableHttpError::Client(McpTransportError::Transport)),
        }
    }

    /// Classify a streaming POST response into an rmcp post response.
    ///
    /// For the `200 text/event-stream` forward path, the body is moved into the
    /// SSE adapter and ownership transfers to the returned stream. For every other
    /// classification (ack, error, buffered JSON), the body is cancelled or drained
    /// before the result is returned.
    #[expect(
        clippy::too_many_lines,
        reason = "response classification mirrors the rmcp reference client"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "response + budget args match the buffered ladder"
    )]
    async fn classify_streaming_post_response(
        &self,
        response: SubResponse,
        mut body: Box<dyn StreamingResponseBody>,
        session_was_attached: bool,
        per_event_cap: usize,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<McpTransportError>> {
        let status = StatusCode::from_u16(response.status)
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;

        if status == StatusCode::UNAUTHORIZED
            && let Some(header) = www_authenticate(&response.headers)
        {
            body.cancel().await;
            return Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(header)));
        }
        if status == StatusCode::FORBIDDEN
            && let Some(header) = www_authenticate(&response.headers)
        {
            body.cancel().await;
            return Err(StreamableHttpError::InsufficientScope(InsufficientScopeError::new(
                header, None,
            )));
        }
        if matches!(status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
            body.cancel().await;
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == StatusCode::NOT_FOUND && session_was_attached {
            body.cancel().await;
            return Err(StreamableHttpError::SessionExpired);
        }

        let session_id_out = response
            .headers
            .get(session_id_header())
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let content_type = response
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);

        if !status.is_success() {
            // Streaming path: drain and fail without parsing a JSON-RPC error. The
            // buffered path extracts a structured error from response.body, but here
            // that would require buffering the entire response just to classify the
            // failure — fail fast on the HTTP status instead.
            drain_body(&mut body).await;
            return Err(StreamableHttpError::UnexpectedServerResponse(
                format!("HTTP {status}").into(),
            ));
        }

        match content_type.as_deref() {
            Some(ct) if is_event_stream_content_type(ct) => {
                let sse = crate::mcp_client::sse_adapter::sse_stream_from_body(
                    body,
                    per_event_cap,
                    per_event_cap, // POST cumulative == per-message ceiling (F3: both from response_limit)
                    max_sse_event_size,
                    self.signal_handle(),
                );
                Ok(StreamableHttpPostResponse::Sse(sse, session_id_out))
            },
            Some(ct) if is_json_content_type(ct) => {
                // A streaming JSON body: buffer it (bounded by the executor cap) and
                // parse the terminal message.
                let buffered = collect_body(&mut body).await;
                match serde_json::from_slice::<ServerJsonRpcMessage>(&buffered) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_out)),
                    Err(_error) => Ok(StreamableHttpPostResponse::Accepted),
                }
            },
            other => {
                body.cancel().await;
                Err(StreamableHttpError::UnexpectedContentType(other.map(str::to_owned)))
            },
        }
    }
}

impl StreamableHttpClient for McpSubrequestClient {
    type Error = McpTransportError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<McpTransportError>> {
        let session_was_attached = session_id.is_some();
        let headers = self.build_post_headers(auth_header, custom_headers, session_id.as_ref())?;
        let max_response_bytes = self.response_limit(&message);
        let body =
            serde_json::to_vec(&message).map_err(|_error| StreamableHttpError::Client(McpTransportError::Serialize))?;
        let response =
            Box::pin(self.execute(Method::POST, &uri, Bytes::from(body), headers, max_response_bytes)).await?;
        classify_buffered_post_response(response, &message, session_was_attached)
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<McpTransportError>> {
        let mut headers = build_request_headers(auth_header, custom_headers)?;
        let value = HeaderValue::from_str(&session_id)
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
        headers.insert(session_id_header(), value);

        let response =
            Box::pin(self.execute(Method::DELETE, &uri, Bytes::new(), headers, MAX_CONTROL_RESPONSE_BYTES)).await?;
        let status = StatusCode::from_u16(response.status)
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
        // A server that does not support session deletion is not an error.
        if status == StatusCode::METHOD_NOT_ALLOWED {
            return Ok(());
        }
        if status.is_success() {
            Ok(())
        } else {
            Err(StreamableHttpError::UnexpectedServerResponse(
                format!("HTTP {status}").into(),
            ))
        }
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<McpTransportError>> {
        self.get_stream_with_max_sse_event_size(
            uri,
            session_id,
            last_event_id,
            auth_header,
            custom_headers,
            DEFAULT_MAX_SSE_EVENT_SIZE,
        )
        .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<McpTransportError>> {
        let headers = self.build_get_stream_headers(
            auth_header,
            custom_headers,
            session_id.as_ref(),
            last_event_id.as_deref(),
        )?;
        let (response, body) = self
            .execute_streaming(
                Method::GET,
                &uri,
                Bytes::new(),
                headers,
                streaming_executor_backstop(self.stream_cumulative_cap()),
            )
            .await?;
        self.classify_get_stream_response(response, body, max_sse_event_size)
            .await
    }

    #[expect(
        clippy::too_many_lines,
        reason = "streaming + buffered-fallback ladder mirrors post_message structure"
    )]
    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<McpTransportError>> {
        // Only client Requests can produce a streaming SSE result worth forwarding
        // incrementally. Notifications / responses / errors are one-way; keep them
        // on the unchanged buffered ack ladder.
        if !matches!(message, ClientJsonRpcMessage::Request(_)) {
            return self
                .post_message(uri, message, session_id, auth_header, custom_headers)
                .await;
        }

        let session_was_attached = session_id.is_some();
        let headers = self.build_post_headers(auth_header, custom_headers, session_id.as_ref())?;
        let max_response_bytes = self.response_limit(&message);
        let body =
            serde_json::to_vec(&message).map_err(|_error| StreamableHttpError::Client(McpTransportError::Serialize))?;

        let (response, maybe_body) = self
            .execute_streaming(
                Method::POST,
                &uri,
                Bytes::from(body),
                headers,
                streaming_executor_backstop(max_response_bytes),
            )
            .await?;
        match maybe_body {
            // Blocker 5: praxis buffered anyway; classify the full buffered response.
            None => classify_buffered_post_response(response, &message, session_was_attached),
            Some(streaming_body) => {
                self.classify_streaming_post_response(
                    response,
                    streaming_body,
                    session_was_attached,
                    max_response_bytes,
                    max_sse_event_size,
                )
                .await
            },
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Classify a fully buffered POST response into an rmcp post response.
/// Verbatim the ladder previously inlined in `post_message`.
#[expect(
    clippy::too_many_lines,
    reason = "response classification mirrors the rmcp reference client"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "takes ownership to match the buffered execute return and the Blocker-5 fallback"
)]
fn classify_buffered_post_response(
    response: SubResponse,
    message: &ClientJsonRpcMessage,
    session_was_attached: bool,
) -> Result<StreamableHttpPostResponse, StreamableHttpError<McpTransportError>> {
    let status = StatusCode::from_u16(response.status)
        .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;

    if status == StatusCode::UNAUTHORIZED
        && let Some(header) = www_authenticate(&response.headers)
    {
        return Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(header)));
    }
    if status == StatusCode::FORBIDDEN
        && let Some(header) = www_authenticate(&response.headers)
    {
        return Err(StreamableHttpError::InsufficientScope(InsufficientScopeError::new(
            header, None,
        )));
    }
    if matches!(status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
        return Ok(StreamableHttpPostResponse::Accepted);
    }
    if status == StatusCode::NOT_FOUND && session_was_attached {
        return Err(StreamableHttpError::SessionExpired);
    }

    let session_id_out = response
        .headers
        .get(session_id_header())
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let content_type = response
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);

    // A framing-only success for a one-way client message is an ack.
    if status.is_success()
        && response.body.is_empty()
        && matches!(
            message,
            ClientJsonRpcMessage::Notification(_) | ClientJsonRpcMessage::Response(_) | ClientJsonRpcMessage::Error(_)
        )
    {
        return Ok(StreamableHttpPostResponse::Accepted);
    }

    if !status.is_success() {
        if content_type.as_deref().is_some_and(is_json_content_type) {
            let body = String::from_utf8_lossy(&response.body);
            if let Some(message) = parse_json_rpc_error(&body) {
                return Ok(StreamableHttpPostResponse::Json(message, session_id_out));
            }
        }
        return Err(StreamableHttpError::UnexpectedServerResponse(
            format!("HTTP {status}").into(),
        ));
    }

    match content_type.as_deref() {
        Some(content_type) if is_event_stream_content_type(content_type) => {
            match parse_buffered_sse_terminal(&response.body) {
                Some(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_out)),
                None => Err(StreamableHttpError::UnexpectedServerResponse(
                    "buffered SSE stream contained no JSON-RPC message".into(),
                )),
            }
        },
        Some(content_type) if is_json_content_type(content_type) => {
            match serde_json::from_slice::<ServerJsonRpcMessage>(&response.body) {
                Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_out)),
                Err(_error) => Ok(StreamableHttpPostResponse::Accepted),
            }
        },
        other => Err(StreamableHttpError::UnexpectedContentType(other.map(str::to_owned))),
    }
}

/// Drain a streaming body to completion, discarding all chunks.
async fn drain_body(body: &mut Box<dyn StreamingResponseBody>) {
    while let Ok(Some(_chunk)) = body.next_chunk().await {}
    body.cancel().await;
}

/// Collect a streaming body into a single `Bytes` buffer.
async fn collect_body(body: &mut Box<dyn StreamingResponseBody>) -> Bytes {
    let mut buf = bytes::BytesMut::new();
    while let Ok(Some(chunk)) = body.next_chunk().await {
        buf.extend_from_slice(&chunk);
    }
    body.cancel().await;
    buf.freeze()
}

/// Merge caller custom headers and an optional bearer token into a header map.
///
/// Reserved Streamable-HTTP headers (`Accept`, `Mcp-Session-Id`,
/// `Last-Event-ID`) are rejected before insertion: the transport sets them from
/// trusted state (content negotiation, the `rmcp`-owned session token, and — for
/// GET resumption — the last event id), so a caller-supplied value could
/// otherwise select or fix a session or override content negotiation. This
/// mirrors `rmcp`'s own reqwest transport, which fails closed on the same set.
///
/// # Errors
///
/// Returns [`StreamableHttpError::ReservedHeaderConflict`] if a caller header is
/// reserved, or [`StreamableHttpError::Client`] with
/// [`McpTransportError::InvalidStatus`] if the auth token is not a valid header
/// value.
fn build_request_headers(
    auth_header: Option<String>,
    custom_headers: HashMap<HeaderName, HeaderValue>,
) -> Result<HeaderMap, StreamableHttpError<McpTransportError>> {
    let mut headers = HeaderMap::new();
    for (name, value) in custom_headers {
        if is_reserved_streamable_http_header(&name) {
            return Err(StreamableHttpError::ReservedHeaderConflict(name.to_string()));
        }
        headers.insert(name, value);
    }
    if let Some(token) = auth_header {
        let value = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
        headers.insert(http::header::AUTHORIZATION, value);
    }
    Ok(headers)
}

/// Whether `name` is a reserved Streamable-HTTP header the transport owns.
///
/// Matches `Accept`, `Mcp-Session-Id`, and `Last-Event-ID` case-insensitively.
/// `MCP-Protocol-Version` is deliberately *not* reserved: callers may pin it.
fn is_reserved_streamable_http_header(name: &HeaderName) -> bool {
    name == http::header::ACCEPT
        || name.as_str().eq_ignore_ascii_case(HEADER_SESSION_ID)
        || name.as_str().eq_ignore_ascii_case(HEADER_LAST_EVENT_ID)
}

/// Extract a `WWW-Authenticate` header value, if present and printable.
fn www_authenticate(headers: &HeaderMap) -> Option<String> {
    headers
        .get(http::header::WWW_AUTHENTICATE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// Compute the origin-form request target (path + query) for `url`.
///
/// # Errors
///
/// Returns [`McpTransportError::Target`] if the URL or its path/query cannot be
/// parsed into a URI.
fn origin_form(url: &str) -> Result<http::Uri, McpTransportError> {
    let uri: http::Uri = url.parse().map_err(|_error| McpTransportError::Target)?;
    let path_and_query = uri.path_and_query().map_or("/", http::uri::PathAndQuery::as_str);
    path_and_query.parse().map_err(|_error| McpTransportError::Target)
}

/// Read back a recorded transport signal as a caller-facing [`McpClientError`],
/// if one was observed on the callout.
///
/// `rmcp` discards the typed [`McpTransportError`] the transport returns, so a
/// [`CalloutOutcome::ResponseTooLarge`] classification or an SSRF address
/// rejection is recorded out-of-band in the client's signal [`OnceLock`] (see
/// [`McpSubrequestClient::signal_handle`]) and read back here after the rmcp
/// `serve`/pagination call fails. Returns [`None`] when no signal was recorded,
/// so the caller can fall back to its generic connection error. `url` carries the
/// credential-safe display URL for the surfaced error.
pub(crate) fn transport_signal_error(
    signal: &OnceLock<TransportSignal>,
    url: &McpDisplayUrl,
) -> Option<McpClientError> {
    match signal.get()? {
        TransportSignal::ResponseTooLarge { limit } => Some(McpClientError::ResponseTooLarge {
            url: url.clone(),
            limit: *limit,
        }),
        TransportSignal::SsrfBlocked => Some(super::ssrf_blocked(url.clone(), super::SSRF_BLOCK_REASON)),
        TransportSignal::TargetRejected => Some(McpClientError::InvalidTarget { url: url.clone() }),
    }
}

/// Validate an MCP target URL against scheme, credential, and SSRF policy on the
/// cache-hit path, where no dial is made.
///
/// On a cache miss the resolved addresses are validated during the actual
/// callout by [`McpSubrequestClient::execute`] (which runs the same
/// [`prepare_url_target`] + [`ssrf_validate`] pair), so this is the *only* extra
/// DNS resolution the resolver performs — and only when a cached listing lets it
/// skip the dial entirely. Reusing [`prepare_url_target`] makes a cache hit
/// enforce exactly the policy a live callout would: scheme/userinfo/fragment
/// rejection, DNS resolution, and the [`ssrf_validate`] address hook.
///
/// # Errors
///
/// Classifies the preparation failure exactly as the live callout would (see
/// [`classify_prepare_error`]): [`McpClientError::SsrfBlocked`] when the target
/// resolves to an address the SSRF policy refuses, [`McpClientError::InvalidTarget`]
/// for a structurally invalid or disallowed URL (bad scheme, embedded userinfo,
/// a fragment, or a malformed/disallowed host literal), or
/// [`McpClientError::Connection`] for a transient failure (DNS resolution or a
/// deadline). The URL is reduced to a credential-safe display form first.
pub(crate) async fn validate_mcp_target(
    url: &str,
    timeout: Duration,
    allow_private: bool,
) -> Result<(), McpClientError> {
    let display_url = super::parse_display_url(url);
    let Some(deadline) = Instant::now().checked_add(timeout) else {
        return Err(McpClientError::Connection { url: display_url });
    };
    match prepare_url_target(url, deadline, move |addrs| ssrf_validate(addrs, allow_private)).await {
        Ok(_target) => Ok(()),
        Err(error) => Err(classify_prepare_error(&error, display_url)),
    }
}

/// Classify a [`prepare_url_target`] failure into the caller-facing
/// [`McpClientError`], distinguishing permanent target rejections that must fail
/// hard from transient dial-time failures.
///
/// Permanent rejections — an SSRF policy block ([`UrlTargetError::PolicyRejected`])
/// and a structurally invalid or disallowed URL ([`UrlTargetError::InvalidTarget`],
/// which also covers host literals rejected at parse time such as a bracketed IPv4
/// or an IPv4-mapped IPv6 address) — map to [`McpClientError::SsrfBlocked`] and
/// [`McpClientError::InvalidTarget`] respectively, so a streaming `tools/list`
/// retains its HTTP error. Transient failures (DNS resolution, a deadline, or any
/// future [`UrlTargetError`] variant) map to [`McpClientError::Connection`], which
/// a streaming listing may surface as a soft in-band lifecycle event.
///
/// This is the single source of truth shared by the cache-hit validator
/// ([`validate_mcp_target`]) and the live callout path ([`prepare_error_signal`]
/// plus [`transport_signal_error`]), guaranteeing identical failure shapes across
/// both.
fn classify_prepare_error(error: &UrlTargetError, url: McpDisplayUrl) -> McpClientError {
    match error {
        UrlTargetError::PolicyRejected(_) => super::ssrf_blocked(url, super::SSRF_BLOCK_REASON),
        UrlTargetError::InvalidTarget(_) => McpClientError::InvalidTarget { url },
        _ => McpClientError::Connection { url },
    }
}

/// Record the out-of-band [`TransportSignal`] for a [`prepare_url_target`]
/// failure, if the failure is a permanent hard rejection.
///
/// Mirrors the permanent/transient split in [`classify_prepare_error`]: an SSRF
/// policy block and a structurally invalid target are recorded so the caller can
/// reconstruct the typed hard rejection after rmcp discards the transport error;
/// transient failures (DNS, deadline) record nothing and fall back to the caller's
/// generic [`McpClientError::Connection`].
fn prepare_error_signal(error: &UrlTargetError) -> Option<TransportSignal> {
    match error {
        UrlTargetError::PolicyRejected(_) => Some(TransportSignal::SsrfBlocked),
        UrlTargetError::InvalidTarget(_) => Some(TransportSignal::TargetRejected),
        _ => None,
    }
}

/// SSRF policy hook applied to the resolved MCP addresses.
///
/// Delegates to [`super::is_ssrf_blocked_ip`] so this hook enforces exactly the
/// policy the literal-IP and DNS-resolution paths do: link-local, unspecified,
/// cloud-metadata, and IPv6 unique-local addresses are always rejected, while
/// loopback and the RFC1918/CGNAT/`0.0.0.0/8` private ranges are rejected unless
/// `allow_private` is set. Sharing the helper closes the gap where an RFC1918
/// address slipped past this hook with private upstreams disabled.
///
/// # Errors
///
/// Returns an opaque, credential-free error when any address is SSRF-sensitive.
fn ssrf_validate(addrs: &[SocketAddr], allow_private: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for addr in addrs {
        if super::is_ssrf_blocked_ip(&addr.ip(), allow_private) {
            return Err(super::SSRF_BLOCK_REASON.into());
        }
    }
    Ok(())
}

/// Map a target-preparation failure to a credential-safe transport error.
fn map_prepare_error(error: &UrlTargetError) -> StreamableHttpError<McpTransportError> {
    match error {
        UrlTargetError::PolicyRejected(_) => StreamableHttpError::Client(McpTransportError::SsrfBlocked),
        _ => StreamableHttpError::Client(McpTransportError::Target),
    }
}

/// Whether a `Content-Type` value denotes JSON.
fn is_json_content_type(value: &str) -> bool {
    value.trim_start().starts_with("application/json")
}

/// Whether a `Content-Type` value denotes an SSE event stream.
fn is_event_stream_content_type(value: &str) -> bool {
    value.trim_start().starts_with("text/event-stream")
}

/// Parse a JSON-RPC error message from a response body, if it is one.
fn parse_json_rpc_error(body: &str) -> Option<ServerJsonRpcMessage> {
    match serde_json::from_str::<ServerJsonRpcMessage>(body) {
        Ok(message @ JsonRpcMessage::Error(_)) => Some(message),
        _ => None,
    }
}

/// Collect the concatenated `data:` payload of each SSE event in `text`.
///
/// Follows the SSE framing rules the transport needs: `data:` lines within an
/// event are joined with `\n`, and a blank line terminates the event.
fn sse_data_events(text: &str) -> Vec<String> {
    let mut events = Vec::new();
    let mut data = String::new();
    let mut has_data = false;
    for line in text.lines() {
        if line.is_empty() {
            if has_data {
                events.push(std::mem::take(&mut data));
                has_data = false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if has_data {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            has_data = true;
        }
    }
    if has_data {
        events.push(data);
    }
    events
}

/// Reparse a buffered SSE body to its terminal JSON-RPC message.
///
/// Returns the first `Response`/`Error` message (the terminal answer for a
/// request/response exchange), falling back to the last parseable message.
fn parse_buffered_sse_terminal(body: &[u8]) -> Option<ServerJsonRpcMessage> {
    let text = std::str::from_utf8(body).ok()?;
    let mut last = None;
    for data in sse_data_events(text) {
        let Ok(message) = serde_json::from_str::<ServerJsonRpcMessage>(&data) else {
            continue;
        };
        if matches!(message, JsonRpcMessage::Response(_) | JsonRpcMessage::Error(_)) {
            return Some(message);
        }
        last = Some(message);
    }
    last
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::net::SocketAddr;

    use super::*;

    // -- Reserved-header hygiene (codex finding: reserved MCP session header) ---

    #[test]
    fn reserved_streamable_http_headers_are_recognized_case_insensitively() {
        assert!(is_reserved_streamable_http_header(&http::header::ACCEPT));
        assert!(is_reserved_streamable_http_header(&HeaderName::from_static(
            "mcp-session-id"
        )));
        assert!(is_reserved_streamable_http_header(&HeaderName::from_static(
            "last-event-id"
        )));
        // MCP-Protocol-Version is intentionally caller-settable, not reserved.
        assert!(!is_reserved_streamable_http_header(&HeaderName::from_static(
            "mcp-protocol-version"
        )));
        assert!(!is_reserved_streamable_http_header(&HeaderName::from_static(
            "x-tenant"
        )));
    }

    #[test]
    fn build_request_headers_rejects_a_caller_supplied_session_id() {
        let mut custom = HashMap::new();
        custom.insert(
            HeaderName::from_static("mcp-session-id"),
            HeaderValue::from_static("attacker-fixed"),
        );
        let error = build_request_headers(None, custom).expect_err("reserved header must be rejected");
        assert!(
            matches!(error, StreamableHttpError::ReservedHeaderConflict(name) if name.eq_ignore_ascii_case("mcp-session-id")),
            "expected a ReservedHeaderConflict for mcp-session-id"
        );
    }

    #[test]
    fn build_request_headers_keeps_allowed_headers_and_injects_bearer() {
        let mut custom = HashMap::new();
        custom.insert(HeaderName::from_static("x-tenant"), HeaderValue::from_static("acme"));
        let headers = build_request_headers(Some("secret-token".to_owned()), custom).expect("allowed headers");
        assert_eq!(headers.get("x-tenant").unwrap(), "acme");
        assert_eq!(headers.get(http::header::AUTHORIZATION).unwrap(), "Bearer secret-token");
    }

    #[test]
    fn build_request_headers_without_auth_sets_no_authorization() {
        let headers = build_request_headers(None, HashMap::new()).expect("no headers");
        assert!(headers.get(http::header::AUTHORIZATION).is_none());
    }

    // -- SSE reparse (codex finding: SSE behavior untested by JSON path) --------

    #[test]
    fn sse_data_events_joins_multiline_data_and_splits_on_blank_lines() {
        let text = "data: line-one\ndata: line-two\n\ndata:second-event\n\n";
        assert_eq!(sse_data_events(text), vec!["line-one\nline-two", "second-event"]);
    }

    #[test]
    fn sse_data_events_emits_trailing_event_without_terminating_blank_line() {
        assert_eq!(sse_data_events("data: only\n"), vec!["only"]);
    }

    #[test]
    fn parse_buffered_sse_terminal_returns_first_response_message() {
        let body = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n",
        );
        let message = parse_buffered_sse_terminal(body.as_bytes()).expect("terminal response");
        assert!(matches!(message, JsonRpcMessage::Response(_)));
    }

    #[test]
    fn parse_buffered_sse_terminal_returns_error_message_as_terminal() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32000,\"message\":\"boom\"}}\n\n";
        let message = parse_buffered_sse_terminal(body.as_bytes()).expect("terminal error");
        assert!(matches!(message, JsonRpcMessage::Error(_)));
    }

    #[test]
    fn parse_buffered_sse_terminal_without_json_is_none() {
        assert!(parse_buffered_sse_terminal(b"data: not-json\n\n").is_none());
        assert!(parse_buffered_sse_terminal(b"").is_none());
    }

    // -- Content-type classification -------------------------------------------

    #[test]
    fn content_type_classification_tolerates_parameters_and_whitespace() {
        assert!(is_json_content_type("application/json"));
        assert!(is_json_content_type(" application/json; charset=utf-8"));
        assert!(!is_json_content_type("text/event-stream"));
        assert!(is_event_stream_content_type("text/event-stream"));
        assert!(is_event_stream_content_type(" text/event-stream; charset=utf-8"));
        assert!(!is_event_stream_content_type("application/json"));
    }

    // -- Response-limit tiers (codex finding: response-limit regressions) -------

    #[test]
    fn tool_result_wire_cap_expands_and_saturates() {
        assert_eq!(
            tool_result_wire_cap(2048),
            2048 * MAX_JSON_STRING_EXPANSION + MAX_TOOL_RESULT_ENVELOPE_BYTES
        );
        // Overflow saturates rather than wrapping to a tiny ceiling.
        assert_eq!(tool_result_wire_cap(usize::MAX), usize::MAX);
    }

    #[test]
    fn streaming_executor_backstop_doubles_and_saturates() {
        assert_eq!(streaming_executor_backstop(1000), 2000);
        assert_eq!(streaming_executor_backstop(usize::MAX), usize::MAX);
    }

    #[test]
    fn response_limit_uses_tool_cap_only_for_tools_call() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            2048,
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .expect("deserialize tools/call");
        assert_eq!(client.response_limit(&call), tool_result_wire_cap(2048));

        let list: ClientJsonRpcMessage =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#)
                .expect("deserialize tools/list");
        assert_eq!(client.response_limit(&list), MAX_CONTROL_RESPONSE_BYTES);
    }

    #[test]
    fn control_client_never_exceeds_the_control_ceiling() {
        let client = McpSubrequestClient::control(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .expect("deserialize tools/call");
        assert_eq!(client.response_limit(&call), MAX_CONTROL_RESPONSE_BYTES);
    }

    // -- SSRF hook (codex finding: chain propagates real posture) --------------

    fn addr(literal: &str) -> SocketAddr {
        literal.parse().expect("socket addr")
    }

    #[test]
    fn ssrf_validate_gates_loopback_on_allow_private() {
        assert!(ssrf_validate(&[addr("127.0.0.1:443")], true).is_ok());
        assert!(ssrf_validate(&[addr("127.0.0.1:443")], false).is_err());
        assert!(ssrf_validate(&[addr("[::1]:443")], true).is_ok());
        assert!(ssrf_validate(&[addr("[::1]:443")], false).is_err());
    }

    #[test]
    fn ssrf_validate_always_blocks_metadata_and_link_local_even_when_private_allowed() {
        assert!(ssrf_validate(&[addr("169.254.169.254:80")], true).is_err());
        assert!(ssrf_validate(&[addr("169.254.1.1:80")], true).is_err());
    }

    #[test]
    fn ssrf_validate_allows_public_addresses() {
        assert!(ssrf_validate(&[addr("93.184.216.34:443")], false).is_ok());
    }

    #[test]
    fn ssrf_validate_rejects_when_any_address_is_sensitive() {
        // A resolved set mixing a public and a metadata address must fail closed.
        assert!(ssrf_validate(&[addr("93.184.216.34:443"), addr("169.254.169.254:80")], true).is_err());
    }

    #[test]
    fn ssrf_validate_gates_rfc1918_ranges_on_allow_private() {
        // Codex finding (RFC1918 SSRF bypass): a pinned private literal must be
        // refused by ssrf_validate itself when private upstreams are disabled —
        // the executor's resolve_address_checked short-circuits an already
        // resolved address, so this hook is the only gate for pinned literals.
        for private in ["10.0.0.1:443", "172.16.5.4:443", "192.168.1.1:443"] {
            assert!(
                ssrf_validate(&[addr(private)], false).is_err(),
                "{private} must be blocked when private upstreams are disabled"
            );
            assert!(
                ssrf_validate(&[addr(private)], true).is_ok(),
                "{private} must be reachable when private upstreams are enabled"
            );
        }
    }

    #[test]
    fn ssrf_validate_gates_cgnat_and_zero_net_on_allow_private() {
        // CGNAT (100.64.0.0/10) and the 0.0.0.0/8 block are private ranges under
        // praxis_core::connectivity::is_private_ip, so they follow the same gate.
        for private in ["100.64.0.1:443", "0.1.2.3:443"] {
            assert!(
                ssrf_validate(&[addr(private)], false).is_err(),
                "{private} must be blocked when private upstreams are disabled"
            );
            assert!(
                ssrf_validate(&[addr(private)], true).is_ok(),
                "{private} must be reachable when private upstreams are enabled"
            );
        }
    }

    #[test]
    fn ssrf_validate_always_blocks_unspecified_ipv4_even_when_private_allowed() {
        // 0.0.0.0 is unspecified: connect(2) routes it to loopback, so it is
        // refused unconditionally rather than gated on allow_private.
        assert!(ssrf_validate(&[addr("0.0.0.0:443")], true).is_err());
        assert!(ssrf_validate(&[addr("0.0.0.0:443")], false).is_err());
    }

    // -- Callout depth resolution (codex finding: nested depth reset to zero) ---

    #[test]
    fn resolve_callout_depth_prefers_iteration_state_over_header() {
        let mut headers = HeaderMap::new();
        headers.insert(DEPTH_HEADER, HeaderValue::from_static("7"));
        // Inside an IRR step the iterative-router's depth is authoritative and
        // must win over any incoming header value.
        assert_eq!(resolve_callout_depth(Some(3), &headers), 3);
    }

    #[test]
    fn resolve_callout_depth_falls_back_to_reserved_header() {
        let mut headers = HeaderMap::new();
        headers.insert(DEPTH_HEADER, HeaderValue::from_static("4"));
        // Without IterationState, a nested sub-request carries its depth in the
        // reserved x-praxis-iterative-depth header so the callout does not reset
        // to zero and under-report its nesting.
        assert_eq!(resolve_callout_depth(None, &headers), 4);
    }

    #[test]
    fn resolve_callout_depth_defaults_to_zero_for_top_level_request() {
        assert_eq!(resolve_callout_depth(None, &HeaderMap::new()), 0);
    }

    #[test]
    fn resolve_callout_depth_defaults_to_zero_on_unparsable_header() {
        for raw in ["not-a-number", "256", "-1", ""] {
            let mut headers = HeaderMap::new();
            headers.insert(DEPTH_HEADER, HeaderValue::from_str(raw).unwrap());
            assert_eq!(
                resolve_callout_depth(None, &headers),
                0,
                "malformed depth {raw:?} must fall back to 0"
            );
        }
    }

    // -- Origin-form target derivation -----------------------------------------

    #[test]
    fn origin_form_extracts_path_and_query() {
        assert_eq!(
            origin_form("https://mcp.example/mcp?cursor=2").unwrap(),
            "/mcp?cursor=2"
        );
    }

    #[test]
    fn origin_form_defaults_to_root_when_path_absent() {
        assert_eq!(origin_form("https://mcp.example").unwrap(), "/");
    }

    // -- JSON-RPC error extraction ---------------------------------------------

    #[test]
    fn parse_json_rpc_error_returns_only_error_messages() {
        assert!(parse_json_rpc_error(r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"x"}}"#).is_some());
        assert!(parse_json_rpc_error(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#).is_none());
        assert!(parse_json_rpc_error("not json").is_none());
    }

    // -- Streaming selector auto-injection -------------------------------------

    #[test]
    fn selector_entry_names_the_selector_with_no_conditions() {
        let entry = streaming_selector_entry();
        assert_eq!(entry.filter_type, "openai_mcp_streaming_selector");
        assert!(entry.conditions.is_empty());
        assert!(entry.name.is_none());
    }

    #[test]
    fn none_chain_becomes_inline_with_selector_first() {
        let chain = selector_injected_chain(None, "mcp_outbound");
        match chain {
            ChainRef::Inline { name, filters } => {
                assert_eq!(name, "mcp_outbound");
                assert_eq!(filters.len(), 1);
                assert_eq!(filters[0].filter_type, "openai_mcp_streaming_selector");
            },
            other @ ChainRef::Named(_) => panic!("expected inline chain, got {other:?}"),
        }
    }

    #[test]
    fn inline_chain_gets_selector_prepended() {
        let inline: ChainRef = serde_yaml::from_str("name: my_chain\nfilters:\n  - filter: headers\n").unwrap();
        let chain = selector_injected_chain(Some(inline), "unused");
        match chain {
            ChainRef::Inline { filters, .. } => {
                assert_eq!(filters.len(), 2);
                assert_eq!(filters[0].filter_type, "openai_mcp_streaming_selector");
                assert_eq!(filters[1].filter_type, "headers");
            },
            other @ ChainRef::Named(_) => panic!("expected inline chain, got {other:?}"),
        }
    }

    #[test]
    fn named_chain_is_returned_unchanged() {
        let named = ChainRef::Named("shared".to_owned());
        let chain = selector_injected_chain(Some(named), "unused");
        assert!(matches!(chain, ChainRef::Named(n) if n == "shared"));
    }

    // -- Stream caps (cumulative GET backstop, Task 4 / F3) --------------------

    #[test]
    fn control_client_caps_are_control_per_event_and_5mib_cumulative() {
        let client = McpSubrequestClient::control(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(1),
        );
        // per-event GET cap == the client's tool-result wire cap, which for the
        // control client is the 1 MiB control ceiling.
        assert_eq!(client.wire_cap(), MAX_CONTROL_RESPONSE_BYTES);
        // cumulative GET cap == 4 MiB listing budget + 1 MiB control budget = 5 MiB.
        assert_eq!(
            client.stream_cumulative_cap(),
            crate::mcp_client::MAX_LISTING_RESPONSE_BYTES + MAX_CONTROL_RESPONSE_BYTES
        );
        assert_eq!(client.stream_cumulative_cap(), 5 * 1024 * 1024);
    }

    #[test]
    fn tool_client_cumulative_cap_is_wire_cap_plus_control() {
        let max_result_bytes = 2 * 1024 * 1024;
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(1),
            max_result_bytes,
        );
        let expected_wire = tool_result_wire_cap(max_result_bytes);
        assert_eq!(client.wire_cap(), expected_wire);
        assert_eq!(
            client.stream_cumulative_cap(),
            expected_wire + MAX_CONTROL_RESPONSE_BYTES
        );
    }

    // -- Streaming POST path (Task 6) --

    fn sub_response(status: u16, content_type: Option<&str>, body: &'static [u8]) -> SubResponse {
        let mut headers = HeaderMap::new();
        if let Some(ct) = content_type {
            headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_str(ct).unwrap());
        }
        SubResponse {
            status,
            headers,
            body: Bytes::from_static(body),
        }
    }

    fn client() -> McpSubrequestClient {
        McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            1024,
        )
    }

    #[tokio::test]
    async fn streaming_post_forwards_event_stream_as_sse() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body = Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
            [Bytes::from_static(
                b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n",
            )],
            Arc::clone(&cancelled),
        ));
        let response = sub_response(200, Some("text/event-stream"), b"");
        let out = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024)
            .await
            .unwrap();
        assert!(matches!(out, StreamableHttpPostResponse::Sse(_, _)));
        // The body is forwarded to the adapter, not cancelled.
        assert!(!cancelled.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn streaming_post_cancels_body_on_202_ack() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body = Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
            [],
            Arc::clone(&cancelled),
        ));
        let response = sub_response(202, None, b"");
        let out = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024)
            .await
            .unwrap();
        assert!(matches!(out, StreamableHttpPostResponse::Accepted));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "non-forward branch cancels the body"
        );
    }

    #[tokio::test]
    async fn streaming_post_cancels_and_maps_401_to_auth_required() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut response = sub_response(401, None, b"");
        response.headers.insert(
            http::header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"mcp\""),
        );
        let body = Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
            [],
            Arc::clone(&cancelled),
        ));
        let err = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024)
            .await
            .unwrap_err();
        assert!(matches!(err, StreamableHttpError::AuthRequired(_)));
        assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn streaming_post_buffers_and_cancels_json_body() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body = Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
            [Bytes::from_static(br#"{"jsonrpc":"2.0","id":1,"result":{}}"#)],
            Arc::clone(&cancelled),
        ));
        let response = sub_response(200, Some("application/json"), b"");
        let out = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024)
            .await
            .unwrap();
        assert!(matches!(out, StreamableHttpPostResponse::Json(_, _)));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "collect_body must cancel after buffering"
        );
    }

    #[test]
    fn classify_buffered_post_response_parses_json() {
        let message: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "x"}
        }))
        .unwrap();
        let response = sub_response(
            200,
            Some("application/json"),
            br#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
        );
        let out = classify_buffered_post_response(response, &message, false).unwrap();
        assert!(matches!(out, StreamableHttpPostResponse::Json(_, _)));
    }

    // -- GET SSE stream path (Task 7) --

    #[tokio::test]
    async fn get_stream_forwards_event_stream() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> =
            Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
                [Bytes::from_static(b"data: {\"jsonrpc\":\"2.0\"}\n\n")],
                cancelled,
            ));
        let response = sub_response(200, Some("text/event-stream"), b"");
        let stream = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024)
            .await
            .unwrap();
        let mut stream = stream;
        let first = futures::StreamExt::next(&mut stream).await.expect("event").expect("ok");
        assert_eq!(first.data.as_deref(), Some("{\"jsonrpc\":\"2.0\"}"));
    }

    #[tokio::test]
    async fn get_stream_405_is_no_sse_support() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([], Arc::clone(&cancelled)),
        );
        let response = sub_response(405, None, b"");
        let result = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024)
            .await;
        assert!(matches!(result, Err(StreamableHttpError::ServerDoesNotSupportSse)));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "non-forward branch cancels the body"
        );
    }

    #[tokio::test]
    async fn classify_get_stream_none_body_is_server_does_not_support_sse() {
        let response = sub_response(200, Some("text/event-stream"), b"");
        let result = client()
            .classify_get_stream_response(response, None, 16 * 1024 * 1024)
            .await;
        assert!(
            matches!(result, Err(StreamableHttpError::ServerDoesNotSupportSse)),
            "200 + text/event-stream + buffered (None) body has no stream to forward"
        );
    }

    #[test]
    fn get_stream_headers_reject_caller_last_event_id() {
        let mut custom = HashMap::new();
        custom.insert(HeaderName::from_static("last-event-id"), HeaderValue::from_static("42"));
        let err = client().build_get_stream_headers(None, custom, None, None).unwrap_err();
        assert!(matches!(err, StreamableHttpError::ReservedHeaderConflict(_)));
    }

    #[test]
    fn get_stream_headers_set_internal_last_event_id() {
        let headers = client()
            .build_get_stream_headers(None, HashMap::new(), None, Some("99"))
            .unwrap();
        assert_eq!(headers.get(HEADER_LAST_EVENT_ID).unwrap(), "99");
        assert_eq!(headers.get(http::header::ACCEPT).unwrap(), "text/event-stream");
    }
}
