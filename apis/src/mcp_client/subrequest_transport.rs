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
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use bytes::Bytes;
use futures::stream::BoxStream;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use praxis_core::{
    config::ChainRef,
    connectivity::{UrlTargetError, prepare_url_target},
    subrequest::{DEPTH_HEADER, SubRequestClient},
};
use praxis_filter::{
    BodyMode, CalloutOutcome, CalloutResponse, CalloutResponseTooLarge, ChainBindingContext, FilterEntry, FilterError,
    FilterPipeline, FilterRegistry, FilteredSubrequestExecutor, HttpFilterContext, IterationState, RequestExtensions,
    StagedUpstream, StagedUpstreamFallback, StreamingResponseBody, SubRequest, SubResponse, SubrequestRuntime,
    TraceContext,
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
use crate::{StateOwner, callout_target::AddressPolicy};

/// Wire byte ceiling for a control-plane MCP response.
///
/// Bounds the buffered body of `initialize`, each `tools/list` page, and any
/// other non-`tools/call` exchange before deserialization, so an untrusted
/// server cannot exhaust proxy memory with an oversized control response. Ported
/// from the previous reqwest-layer `bounded_http` client; this is now the single
/// home for the MCP response tiers, since the dial runs through the executor.
/// `mod.rs` derives its cumulative `tools/list` budget from this value.
pub(crate) const MAX_CONTROL_RESPONSE_BYTES: usize = 1_048_576;

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

/// Cap a buffered reply before it is materialized. The JSON preparse estimate
/// always includes at least three wire-sized owners, so a larger raw body
/// cannot pass the later parse gate even if its payload is otherwise valid.
fn buffered_preparse_wire_cap(wire_cap: usize, peak_limit: Option<usize>) -> usize {
    wire_cap.min(peak_limit.map_or(usize::MAX, |peak| peak / 3))
}

/// Bound the raw response, decoded strings/numbers, and JSON tree nodes before
/// rmcp deserializes an untrusted tool result. The node allowance covers `Value`
/// storage and container growth even for dense arrays of one-byte values.
#[expect(
    clippy::too_many_lines,
    reason = "one lexical pass counts numeric expansion and JSON nodes"
)]
pub(super) fn json_preparse_peak_bytes(raw: &[u8]) -> Option<usize> {
    // Two wire-like owners may coexist: the transport body and SSE `data` or
    // serde's string allocation. rmcp's untagged JSON-RPC deserializer can
    // hold intermediate Value trees alongside the final one. With serde_json's
    // preserve_order maps, dense arrays of small nested objects need more than
    // 128 bytes per structural token before those intermediates are dropped.
    const NODE_PEAK_BYTES: usize = 512;
    let mut normalized = raw.len();
    let mut nodes = 1_usize;
    let mut index = 0;
    let mut quoted = false;
    let mut escaped = false;
    while let Some(&byte) = raw.get(index) {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'[' | b'{' | b':' | b',' => nodes = nodes.checked_add(1)?,
            b'-' | b'0'..=b'9' => {
                let start = index;
                let mut exponent = false;
                while let Some(&digit) = raw.get(index) {
                    if !matches!(digit, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                        break;
                    }
                    exponent |= matches!(digit, b'e' | b'E');
                    index += 1;
                }
                let token = raw.get(start..index)?;
                if exponent || token.len() >= 20 || token == b"-0" {
                    normalized = normalized.checked_add(24_usize.saturating_sub(token.len()))?;
                }
                continue;
            },
            _ => {},
        }
        index += 1;
    }
    raw.len()
        .checked_mul(2)?
        .checked_add(normalized)?
        .checked_add(nodes.checked_mul(NODE_PEAK_BYTES)?)
}

/// True when the raw message and its worst-case parsed owners fit the optional cap.
pub(super) fn json_preparse_fits(raw: &[u8], limit: Option<usize>) -> bool {
    limit.is_none_or(|limit| json_preparse_peak_bytes(raw).is_some_and(|bytes| bytes <= limit))
}

/// Record a typed size error before rmcp can allocate a parsed JSON-RPC message.
fn admit_json_preparse(
    raw: &[u8],
    limit: Option<usize>,
    signal: &Arc<OnceLock<TransportSignal>>,
) -> Result<(), StreamableHttpError<McpTransportError>> {
    if json_preparse_fits(raw, limit) {
        return Ok(());
    }
    let cap = limit.unwrap_or(0);
    signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit: cap });
    Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
}

/// Loose executor backstop for a streaming callout (spec §4.5, F3).
///
/// praxis arms its outer `CalloutStreamingBody` at the `max_response_bytes`
/// passed to [`McpSubrequestClient::execute_streaming`]. The executor ceiling is
/// twice the binding adapter cap so the adapter normally enforces its exact
/// per-message/cumulative limit first. If one chunk exceeds the outer ceiling,
/// core returns a typed size error and the SSE adapter records the binding cap.
/// The finite outer ceiling also bounds buffered fallback and streaming JSON
/// drains (`collect_body`/`drain_body`) at twice the cap.
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

/// Replaceable transport-error slot for a reusable rmcp session.
///
/// Each exclusive `tools/call` installs a fresh `OnceLock` before sending. The
/// transport clones read the current generation when starting a POST, while the
/// standalone GET SSE stream uses a detached slot. An idle-stream failure can
/// therefore never be mistaken for the later tool call's failure.
pub(crate) struct TransportSignalState {
    /// Signal generation assigned to the active initialization or tool call.
    active: Mutex<Option<Arc<OnceLock<TransportSignal>>>>,
}

impl TransportSignalState {
    /// Create state with a pristine initialization-generation signal.
    fn new() -> Self {
        Self {
            active: Mutex::new(Some(Arc::new(OnceLock::new()))),
        }
    }

    /// Return the active signal, or a detached slot for idle traffic.
    pub(crate) fn current(&self) -> Arc<OnceLock<TransportSignal>> {
        let guard = self.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.as_ref().map_or_else(|| Arc::new(OnceLock::new()), Arc::clone)
    }

    /// Install and return a pristine signal generation for one tool call.
    pub(crate) fn begin_exchange(&self) -> Arc<OnceLock<TransportSignal>> {
        let signal = Arc::new(OnceLock::new());
        let mut guard = self.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = Some(Arc::clone(&signal));
        signal
    }

    /// Clear `signal` only if it still owns the active exchange generation.
    pub(crate) fn finish_exchange(&self, signal: &Arc<OnceLock<TransportSignal>>) {
        let mut guard = self.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.as_ref().is_some_and(|active| Arc::ptr_eq(active, signal)) {
            *guard = None;
        }
    }

    /// Record an SSE failure only while a tool call owns an active generation.
    pub(crate) fn record_active(&self, classification: TransportSignal) {
        let guard = self.active.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(signal) = guard.as_ref() {
            signal.get_or_init(|| classification);
        }
    }
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
/// [`call_tool_with_forwarded_headers_with_budget`](super::call_tool_with_forwarded_headers_with_budget), which build
/// an [`McpSubrequestClient`] from it.
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
    /// Approved request correlation projected into every MCP exchange.
    trace_context: Option<TraceContext>,
    /// Absolute enclosing IRR deadline shared by initialize/list/call exchanges.
    parent_deadline: Option<Instant>,
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
        let iteration_state = ctx.extensions.get::<IterationState>();
        let depth = resolve_callout_depth(iteration_state.map(IterationState::depth), &ctx.request.headers);
        let parent_deadline = iteration_state.map(IterationState::deadline);
        let trace_context = ctx.extensions.get::<TraceContext>().cloned();
        Some(Self {
            client,
            downstream,
            pipeline,
            depth,
            trace_context,
            parent_deadline,
            allow_private,
        })
    }

    /// Whether private/loopback MCP destinations are permitted for this callout.
    pub(crate) fn allow_private(&self) -> bool {
        self.allow_private
    }

    /// Bound one MCP transport exchange by the enclosing IRR deadline.
    fn deadline(&self, now: Instant, step_timeout: Duration) -> Instant {
        let step_deadline = now.checked_add(step_timeout).unwrap_or(now);
        self.parent_deadline
            .map_or(step_deadline, |parent| step_deadline.min(parent))
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
            client: crate::subrequest::isolated_client(1),
            downstream: SubrequestRuntime::new(None, false, None, Instant::now()),
            pipeline,
            depth: 0,
            trace_context: None,
            parent_deadline: None,
            allow_private,
        })
    }

    /// Attach an explicit parent deadline to a fabricated test callout.
    #[cfg(test)]
    pub(crate) fn with_parent_deadline_for_test(mut self, deadline: Instant) -> Self {
        self.parent_deadline = Some(deadline);
        self
    }

    /// Replace the bare test pipeline with an observable one.
    #[cfg(test)]
    pub(crate) fn with_pipeline_for_test(mut self, pipeline: Arc<FilterPipeline>) -> Self {
        self.pipeline = pipeline;
        self
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
    /// `tools/call` responses use this configured, JSON-expansion-adjusted
    /// ceiling (see [`Self::response_limit`]).
    tool_result_bytes: usize,
    /// Optional request-wide allowance for a `tools/call` raw wire plus rmcp's
    /// parsed JSON. Control exchanges have their own admission reservation.
    preparse_peak_limit: Option<usize>,
    /// Per-exchange cap for initialize and other control responses. Budgeted
    /// tool calls lower this before dialing a fresh session.
    control_response_bytes: usize,
    /// Cumulative wire-byte ceiling for a server-initiated GET SSE stream.
    ///
    /// An intentionally coarse raw-wire `DoS` backstop, not decoded parity: for a
    /// control client it is `MAX_LISTING_RESPONSE_BYTES + MAX_CONTROL_RESPONSE_BYTES`
    /// (5 MiB); `paginate_tools` remains the authoritative decoded gate. Read
    /// only by the GET-stream path.
    stream_cumulative_cap: usize,
    /// Per-exchange duration ceiling.
    step_timeout: Duration,
    /// Trusted owner projected only for configured connector exchanges.
    owner: Option<StateOwner>,
    /// Replaceable out-of-band record of a typed classification observed on a
    /// control or tool POST.
    ///
    /// `rmcp` discards the typed [`McpTransportError`] on failure, so a
    /// [`CalloutOutcome::ResponseTooLarge`] classification or an SSRF address
    /// rejection is recorded here (first signal wins) and read back by the caller
    /// via [`transport_signal_error`] after the rmcp `serve`/pagination call
    /// fails. Shared through the [`Clone`] the transport requires, so the handle
    /// taken before the client is moved into the rmcp transport observes writes
    /// made during the exchange.
    signal_state: Arc<TransportSignalState>,
}

impl McpSubrequestClient {
    /// Build a client for the control-plane exchanges performed by
    /// [`list_tools_with_forwarded_headers`](super::list_tools_with_forwarded_headers):
    /// `initialize` and `tools/list`.
    ///
    /// No `tools/call` result flows over this transport, so every response is
    /// bounded to [`MAX_CONTROL_RESPONSE_BYTES`] before deserialization.
    pub(crate) fn control(callout: McpCallout, step_timeout: Duration, owner: Option<StateOwner>) -> Self {
        Self::with_wire_cap(
            callout,
            step_timeout,
            MAX_CONTROL_RESPONSE_BYTES,
            MAX_CONTROL_RESPONSE_BYTES,
            crate::mcp_client::MAX_LISTING_RESPONSE_BYTES.saturating_add(MAX_CONTROL_RESPONSE_BYTES),
            None,
            owner,
        )
    }

    /// Build a client for
    /// [`call_tool_with_forwarded_headers_with_budget`](super::call_tool_with_forwarded_headers_with_budget):
    /// `initialize` uses the supplied control ceiling and the `tools/call`
    /// response uses `max_result_bytes`, expanded for JSON string escaping.
    ///
    /// `step_timeout` bounds each individual HTTP exchange; the `callout` carries
    /// the parent transport and the bound outbound pipeline whose finalized
    /// posture decides whether loopback destinations are permitted.
    #[expect(
        clippy::too_many_arguments,
        reason = "tool transport has independent wire and parse caps"
    )]
    pub(crate) fn for_tool(
        callout: McpCallout,
        step_timeout: Duration,
        max_result_bytes: usize,
        max_control_response_bytes: usize,
        preparse_peak_limit: Option<usize>,
        owner: Option<StateOwner>,
    ) -> Self {
        let wire = tool_result_wire_cap(max_result_bytes);
        Self::with_wire_cap(
            callout,
            step_timeout,
            wire,
            max_control_response_bytes,
            wire.saturating_add(max_control_response_bytes),
            preparse_peak_limit,
            owner,
        )
    }

    /// Shared constructor: move in the callout and pin the `tools/call` wire
    /// ceiling.
    #[expect(
        clippy::too_many_arguments,
        reason = "tool and control wire caps are independent transport bounds"
    )]
    fn with_wire_cap(
        callout: McpCallout,
        step_timeout: Duration,
        tool_result_bytes: usize,
        control_response_bytes: usize,
        stream_cumulative_cap: usize,
        preparse_peak_limit: Option<usize>,
        owner: Option<StateOwner>,
    ) -> Self {
        Self {
            callout,
            tool_result_bytes,
            preparse_peak_limit,
            control_response_bytes,
            step_timeout,
            stream_cumulative_cap,
            owner,
            signal_state: Arc::new(TransportSignalState::new()),
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
        self.signal_state.current()
    }

    /// Shared state used by a pooled session to install a fresh signal for each
    /// exclusive tool call.
    pub(crate) fn signal_state(&self) -> Arc<TransportSignalState> {
        Arc::clone(&self.signal_state)
    }

    /// Per-event wire ceiling for a GET SSE stream (the tool-result wire cap).
    fn wire_cap(&self) -> usize {
        self.tool_result_bytes
    }

    /// Cumulative wire ceiling for a GET SSE stream.
    fn stream_cumulative_cap(&self) -> usize {
        self.stream_cumulative_cap
    }

    /// A non-streaming outbound chain can buffer a GET before the SSE path
    /// rejects it. Apply the parse allowance at that core admission boundary.
    fn buffered_get_response_cap(&self) -> usize {
        buffered_preparse_wire_cap(
            streaming_executor_backstop(self.stream_cumulative_cap()),
            self.preparse_peak_limit,
        )
    }

    /// Select the wire byte ceiling for one outbound message.
    ///
    /// `tools/call` responses use the configured tool-result ceiling; every other
    /// exchange (`initialize`, `tools/list`, notifications, ...) is bounded to the
    /// control ceiling so an untrusted server cannot exhaust proxy memory on the
    /// control plane.
    fn response_limit(&self, message: &ClientJsonRpcMessage) -> usize {
        if Self::is_tool_call(message) {
            self.tool_result_bytes
        } else {
            self.control_response_bytes
        }
    }

    /// Tool replies use the result reservation; control exchanges use their
    /// separately admitted raw and parsed response allowance.
    fn preparse_limit(&self, message: &ClientJsonRpcMessage) -> Option<usize> {
        if Self::is_tool_call(message) {
            self.preparse_peak_limit
        } else {
            self.control_preparse_limit()
        }
    }

    /// Parsed control owners share the same multiplier reserved by dispatch.
    fn control_preparse_limit(&self) -> Option<usize> {
        self.preparse_peak_limit
            .map(|_| self.control_response_bytes * super::MCP_CONTROL_RESPONSE_PEAK_MULTIPLIER)
    }

    /// Match the request whose reply uses the configured tool-result ceiling.
    fn is_tool_call(message: &ClientJsonRpcMessage) -> bool {
        matches!(message, ClientJsonRpcMessage::Request(request) if matches!(request.request, ClientRequest::CallToolRequest(_)))
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
        max_buffered_response_bytes: usize,
        max_streaming_response_bytes: usize,
        signal: &Arc<OnceLock<TransportSignal>>,
    ) -> Result<
        (FilteredSubrequestExecutor, SubRequest, RequestExtensions, Instant),
        StreamableHttpError<McpTransportError>,
    > {
        let deadline = self.callout.deadline(Instant::now(), self.step_timeout);
        let allow_private = self.callout.allow_private;
        let target = match prepare_url_target(uri, deadline, move |addrs| ssrf_validate(addrs, allow_private)).await {
            Ok(target) => target,
            Err(error) => {
                if let Some(classification) = prepare_error_signal(&error) {
                    // First signal wins; the free-standing caller reconstructs the
                    // typed hard rejection (SSRF or invalid target) after rmcp
                    // discards the transport error (see `transport_signal_error`).
                    // Transient failures (DNS, deadline) record nothing and fall
                    // back to the caller's generic connection error.
                    signal.get_or_init(|| classification);
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
        if let Some(owner) = self.owner.as_ref() {
            extensions.insert(owner.clone());
        }
        if let Some(trace_context) = self.callout.trace_context.as_ref() {
            extensions.insert(trace_context.clone());
        }

        let executor = FilteredSubrequestExecutor::for_callout_with_limits(
            self.callout.client.clone(),
            self.callout.downstream.clone(),
            self.callout.depth,
            max_buffered_response_bytes,
            max_streaming_response_bytes,
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
        signal: &Arc<OnceLock<TransportSignal>>,
    ) -> Result<SubResponse, StreamableHttpError<McpTransportError>> {
        let (executor, request, extensions, deadline) = self
            .prepare_staged_request(
                method,
                uri,
                body,
                headers,
                max_response_bytes,
                max_response_bytes,
                signal,
            )
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
                signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
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
    #[expect(
        clippy::too_many_lines,
        reason = "response classification mirrors the rmcp reference client"
    )]
    async fn classify_get_stream_response(
        &self,
        response: SubResponse,
        body: Option<Box<dyn StreamingResponseBody>>,
        max_sse_event_size: usize,
        signal: crate::mcp_client::sse_adapter::SseSignalTarget,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<McpTransportError>> {
        let status = StatusCode::from_u16(response.status)
            .map_err(|_error| StreamableHttpError::Client(McpTransportError::InvalidStatus))?;
        let content_type = response
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok());

        if status == StatusCode::UNAUTHORIZED
            && let Some(header) = www_authenticate(&response.headers)
        {
            if let Some(mut body) = body {
                body.cancel().await;
            }
            return Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(header)));
        }
        if status == StatusCode::FORBIDDEN
            && let Some(header) = www_authenticate(&response.headers)
        {
            if let Some(mut body) = body {
                body.cancel().await;
            }
            return Err(StreamableHttpError::InsufficientScope(InsufficientScopeError::new(
                header, None,
            )));
        }

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
            // rmcp always passes its `config.max_sse_event_size` (16 MiB default),
            // which is only an outer sanity backstop. Raise it to the wire cap so it
            // can never clamp the authoritative per-event bound below `wire_cap()`.
            max_sse_event_size.max(self.wire_cap()),
            signal,
            // rmcp may resume a tool POST response through GET after the SSE
            // connection drops. Preserve the tool reply's parse allowance.
            self.preparse_peak_limit,
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
        max_buffered_response_bytes: usize,
        max_streaming_response_bytes: usize,
        signal: &Arc<OnceLock<TransportSignal>>,
    ) -> Result<(SubResponse, Option<Box<dyn StreamingResponseBody>>), StreamableHttpError<McpTransportError>> {
        let (executor, request, mut extensions, deadline) = self
            .prepare_staged_request(
                method,
                uri,
                body,
                headers,
                max_buffered_response_bytes,
                max_streaming_response_bytes,
                signal,
            )
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
                signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
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
        preparse_peak_limit: Option<usize>,
        signal: Arc<OnceLock<TransportSignal>>,
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
            // Mirror the buffered classifier: surface a structured JSON-RPC error
            // to the client instead of collapsing every failure to a generic
            // "HTTP {status}". Bound the error body by the control ceiling and,
            // unlike the success drain, record NO size signal — an oversize or
            // unparseable error body falls back to the HTTP status, never a
            // spurious 413 (see collect_capped). Exactly one of the two branches
            // cancels the body.
            if content_type.as_deref().is_some_and(is_json_content_type) {
                if preparse_peak_limit.is_some() && !body.try_cap_chunk_bytes(self.control_response_bytes) {
                    body.cancel().await;
                    return Err(StreamableHttpError::UnexpectedServerResponse(
                        format!("HTTP {status}").into(),
                    ));
                }
                if let Some(bytes) = collect_capped(&mut body, self.control_response_bytes).await
                    && json_preparse_fits(&bytes, preparse_peak_limit)
                    && let Ok(text) = std::str::from_utf8(&bytes)
                    && let Some(message) = parse_json_rpc_error(text)
                {
                    return Ok(StreamableHttpPostResponse::Json(message, session_id_out));
                }
            } else {
                drain_body(&mut body).await;
            }
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
                    // rmcp's `config.max_sse_event_size` is an outer backstop only; raise it to
                    // the per-message cap so it never clamps the per-event bound below it.
                    max_sse_event_size.max(per_event_cap),
                    Arc::clone(&signal).into(),
                    preparse_peak_limit,
                );
                Ok(StreamableHttpPostResponse::Sse(sse, session_id_out))
            },
            Some(ct) if is_json_content_type(ct) => {
                // A streaming JSON body: buffer it (bounded by the per-message cap,
                // failing closed on oversize or a transport error) and parse the
                // terminal message. A Request always needs a reply, so an
                // unparseable body is a typed UnexpectedServerResponse, never an
                // Accepted ack (which is reserved for one-way messages).
                if let Some(limit) = preparse_peak_limit
                    && !body.try_cap_chunk_bytes(buffered_preparse_wire_cap(per_event_cap, Some(limit)))
                {
                    signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                    body.cancel().await;
                    return Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge));
                }
                let buffered = collect_body(&mut body, per_event_cap, preparse_peak_limit, &signal).await?;
                admit_json_preparse(&buffered, preparse_peak_limit, &signal)?;
                match serde_json::from_slice::<ServerJsonRpcMessage>(&buffered) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_out)),
                    Err(_error) => Err(StreamableHttpError::UnexpectedServerResponse(
                        "streaming JSON response was not a valid JSON-RPC message".into(),
                    )),
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
        let preparse_peak_limit = self.preparse_limit(&message);
        let buffered_cap = buffered_preparse_wire_cap(max_response_bytes, preparse_peak_limit);
        let signal = self.signal_handle();
        let body =
            serde_json::to_vec(&message).map_err(|_error| StreamableHttpError::Client(McpTransportError::Serialize))?;
        let response =
            Box::pin(self.execute(Method::POST, &uri, Bytes::from(body), headers, buffered_cap, &signal)).await?;
        classify_buffered_post_response(response, &message, session_was_attached, preparse_peak_limit, &signal)
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

        let signal = Arc::new(OnceLock::new());
        let response = Box::pin(self.execute(
            Method::DELETE,
            &uri,
            Bytes::new(),
            headers,
            self.control_response_bytes,
            &signal,
        ))
        .await?;
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
        // Standalone GET failures belong to the idle stream, not the next tool
        // POST. Keep their signal generation detached from the reusable call
        // state so they cannot poison later error classification.
        let signal = Arc::new(OnceLock::new());
        let (response, body) = self
            .execute_streaming(
                Method::GET,
                &uri,
                Bytes::new(),
                headers,
                self.buffered_get_response_cap(),
                streaming_executor_backstop(self.stream_cumulative_cap()),
                &signal,
            )
            .await?;
        self.classify_get_stream_response(
            response,
            body,
            max_sse_event_size,
            crate::mcp_client::sse_adapter::SseSignalTarget::Active(Arc::clone(&self.signal_state)),
        )
        .await
    }

    #[expect(
        clippy::too_many_lines,
        reason = "streaming + buffered-fallback ladder mirrors post_message structure"
    )]
    #[expect(clippy::large_stack_frames, reason = "rmcp/executor futures are inherently large")]
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
        let preparse_peak_limit = self.preparse_limit(&message);
        let signal = self.signal_handle();
        let body =
            serde_json::to_vec(&message).map_err(|_error| StreamableHttpError::Client(McpTransportError::Serialize))?;

        let (response, maybe_body) = self
            .execute_streaming(
                Method::POST,
                &uri,
                Bytes::from(body),
                headers,
                buffered_preparse_wire_cap(max_response_bytes, preparse_peak_limit),
                streaming_executor_backstop(max_response_bytes),
                &signal,
            )
            .await?;
        match maybe_body {
            // Blocker 5: praxis buffered anyway; classify the full buffered response.
            None => {
                classify_buffered_post_response(response, &message, session_was_attached, preparse_peak_limit, &signal)
            },
            Some(streaming_body) => {
                self.classify_streaming_post_response(
                    response,
                    streaming_body,
                    session_was_attached,
                    max_response_bytes,
                    max_sse_event_size,
                    preparse_peak_limit,
                    signal,
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
    preparse_peak_limit: Option<usize>,
    signal: &Arc<OnceLock<TransportSignal>>,
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
        if content_type.as_deref().is_some_and(is_json_content_type)
            && json_preparse_fits(&response.body, preparse_peak_limit)
            && let Ok(text) = std::str::from_utf8(&response.body)
            && let Some(message) = parse_json_rpc_error(text)
        {
            return Ok(StreamableHttpPostResponse::Json(message, session_id_out));
        }
        return Err(StreamableHttpError::UnexpectedServerResponse(
            format!("HTTP {status}").into(),
        ));
    }

    match content_type.as_deref() {
        Some(content_type) if is_event_stream_content_type(content_type) => {
            match parse_buffered_sse_terminal(&response.body, preparse_peak_limit, signal)? {
                Some(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_out)),
                None => Err(StreamableHttpError::UnexpectedServerResponse(
                    "buffered SSE stream contained no JSON-RPC message".into(),
                )),
            }
        },
        Some(content_type) if is_json_content_type(content_type) => {
            admit_json_preparse(&response.body, preparse_peak_limit, signal)?;
            match serde_json::from_slice::<ServerJsonRpcMessage>(&response.body) {
                Ok(message) => Ok(StreamableHttpPostResponse::Json(message, session_id_out)),
                // A Request always needs a reply; an unparseable body is a typed
                // UnexpectedServerResponse, never an Accepted ack (which is
                // reserved for one-way messages).
                Err(_error) => Err(StreamableHttpError::UnexpectedServerResponse(
                    "buffered JSON response was not a valid JSON-RPC message".into(),
                )),
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

/// Collect a streaming body into a single `Bytes` buffer, failing closed.
///
/// Accumulates chunks into one bounded [`bytes::BytesMut`] and enforces a local
/// byte `cap` and a separate peak allowance before growing its buffer. On
/// overflow the size signal is recorded into
/// `signal` (first wins) so the typed 413 survives rmcp's opaque error mapping,
/// the body is cancelled, and a typed [`McpTransportError::ResponseTooLarge`] is
/// returned — never the truncated buffer.
///
/// A `next_chunk()` error also fails closed the same way. The caller tightens
/// the streaming body's per-chunk cap before this drain; praxis withholds an
/// oversize chunk as a typed error. A rare unrelated read error also becomes
/// `ResponseTooLarge` on this size-bounded path. Clean EOF (`Ok(None)`) cancels
/// the body and returns the buffered bytes.
#[expect(
    clippy::too_many_lines,
    reason = "buffer admission and cancellation share one body loop"
)]
async fn collect_body(
    body: &mut Box<dyn StreamingResponseBody>,
    cap: usize,
    preparse_peak_limit: Option<usize>,
    signal: &Arc<OnceLock<TransportSignal>>,
) -> Result<Bytes, StreamableHttpError<McpTransportError>> {
    let mut buf = bytes::BytesMut::new();
    loop {
        match body.next_chunk().await {
            Ok(Some(chunk)) => {
                let next_len = buf.len().checked_add(chunk.len());
                if next_len.is_none_or(|bytes| bytes > cap) {
                    signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit: cap });
                    body.cancel().await;
                    return Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge));
                }
                if let Some(limit) = preparse_peak_limit {
                    // BytesMut may replace its backing allocation when it grows.
                    // Keep the incoming chunk, old capacity, and at most twice
                    // the new logical length inside the parse allowance.
                    let growth_peak = next_len.and_then(|next| {
                        let live = buf.capacity().checked_add(chunk.len())?;
                        if next > buf.capacity() {
                            live.checked_add(next.checked_mul(2)?.max(8))
                        } else {
                            Some(live)
                        }
                    });
                    if growth_peak.is_none_or(|bytes| bytes > limit) {
                        signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                        body.cancel().await;
                        return Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge));
                    }
                }
                buf.extend_from_slice(&chunk);
            },
            Ok(None) => {
                body.cancel().await;
                return Ok(buf.freeze());
            },
            Err(error) => {
                // The armed streaming body withholds an oversize chunk before
                // delivery. Preserve its typed limit when present; other read
                // errors still fail closed as a 413 on this size-bounded drain.
                let limit = error
                    .downcast_ref::<CalloutResponseTooLarge>()
                    .map_or(cap, |error| error.limit);
                signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                body.cancel().await;
                return Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge));
            },
        }
    }
}

/// Collect a non-2xx error body into a bounded buffer for structured-error
/// extraction, without recording a size signal.
///
/// Distinct from [`collect_body`] on purpose: on the error path an oversize or
/// truncated body must NOT surface as a 413. This helper takes no signal handle —
/// so it structurally cannot record [`TransportSignal::ResponseTooLarge`] — and
/// returns [`None`] on overflow (`buf.len() > cap`) or a `next_chunk()` error,
/// letting the caller fall back to the true HTTP status. The body is cancelled in
/// every case; a clean EOF within `cap` returns `Some(bytes)`.
async fn collect_capped(body: &mut Box<dyn StreamingResponseBody>, cap: usize) -> Option<Bytes> {
    let mut buf = bytes::BytesMut::new();
    loop {
        match body.next_chunk().await {
            Ok(Some(chunk)) => {
                if buf.len().checked_add(chunk.len()).is_none_or(|bytes| bytes > cap) {
                    body.cancel().await;
                    return None;
                }
                buf.extend_from_slice(&chunk);
            },
            Ok(None) => {
                body.cancel().await;
                return Some(buf.freeze());
            },
            Err(_error) => {
                body.cancel().await;
                return None;
            },
        }
    }
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
/// Uses the shared AI address policy for both literal-IP and DNS-resolution
/// paths. Metadata, unspecified, and multicast destinations remain forbidden;
/// other non-public destinations follow the explicit private-target opt-in.
///
/// # Errors
///
/// Returns an opaque, credential-free error when any address is SSRF-sensitive.
fn ssrf_validate(addrs: &[SocketAddr], allow_private: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let policy = AddressPolicy::from_allow_private(allow_private);
    for addr in addrs {
        if policy.blocks(&addr.ip()) {
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
#[cfg(test)]
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
#[expect(
    clippy::too_many_lines,
    reason = "checks each SSE event's live parse owners before allocation"
)]
fn parse_buffered_sse_terminal(
    body: &[u8],
    preparse_peak_limit: Option<usize>,
    signal: &Arc<OnceLock<TransportSignal>>,
) -> Result<Option<ServerJsonRpcMessage>, StreamableHttpError<McpTransportError>> {
    let Ok(text) = std::str::from_utf8(body) else {
        return Ok(None);
    };
    let mut last = None;
    let mut last_charge = 0_usize;
    let mut data = String::new();
    let mut has_data = false;
    // Parse one SSE event at a time. The bounded transport body remains live,
    // so account for it and the prior fallback message before growing data or
    // allocating a JSON-RPC object.
    for line in text.lines().chain(std::iter::once("")) {
        if line.is_empty() {
            if !has_data {
                continue;
            }
            let current = std::mem::take(&mut data);
            has_data = false;
            let current_charge = if let Some(limit) = preparse_peak_limit {
                let peak = json_preparse_peak_bytes(current.as_bytes());
                if peak
                    .and_then(|bytes| body.len().checked_add(bytes))
                    .and_then(|bytes| bytes.checked_add(last_charge))
                    .is_none_or(|bytes| bytes > limit)
                {
                    signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                    return Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge));
                }
                peak.unwrap_or(usize::MAX)
            } else {
                0
            };
            let Ok(message) = serde_json::from_str::<ServerJsonRpcMessage>(&current) else {
                continue;
            };
            if matches!(message, JsonRpcMessage::Response(_) | JsonRpcMessage::Error(_)) {
                return Ok(Some(message));
            }
            last_charge = current_charge;
            last = Some(message);
            continue;
        }
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        let next_len = data
            .len()
            .checked_add(usize::from(has_data))
            .and_then(|bytes| bytes.checked_add(rest.len()));
        // String growth can temporarily retain the old allocation alongside
        // a doubled replacement while the wire body lives.
        if let Some(limit) = preparse_peak_limit
            && next_len
                .and_then(|bytes| bytes.checked_mul(3))
                .and_then(|bytes| bytes.checked_add(body.len()))
                .and_then(|bytes| bytes.checked_add(last_charge))
                .is_none_or(|bytes| bytes > limit)
        {
            signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
            return Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge));
        }
        if has_data {
            data.push('\n');
        }
        data.push_str(rest);
        has_data = true;
    }
    Ok(last)
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

    use http::HeaderValue;
    use praxis_filter::builtins::TraceContextFilter;

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    fn test_signal() -> Arc<OnceLock<TransportSignal>> {
        Arc::new(OnceLock::new())
    }

    #[test]
    fn reusable_session_starts_each_exchange_with_a_pristine_signal() {
        let state = TransportSignalState::new();
        let prior = state.current();
        assert!(prior.set(TransportSignal::ResponseTooLarge { limit: 7 }).is_ok());

        let current = state.begin_exchange();
        assert!(
            current.get().is_none(),
            "a later call must not inherit an idle-stream or prior-call error"
        );
        assert!(
            matches!(prior.get(), Some(TransportSignal::ResponseTooLarge { limit: 7 })),
            "replacing the current generation must not mutate in-flight readers"
        );

        state.finish_exchange(&current);
        state.record_active(TransportSignal::SsrfBlocked);
        assert!(
            current.get().is_none(),
            "idle GET failures must not poison the completed call"
        );

        let next = state.begin_exchange();
        state.record_active(TransportSignal::TargetRejected);
        assert!(
            matches!(next.get(), Some(TransportSignal::TargetRejected)),
            "GET failures during a call must reach that active generation"
        );
    }

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
        let message = parse_buffered_sse_terminal(body.as_bytes(), None, &test_signal())
            .unwrap()
            .expect("terminal response");
        assert!(matches!(message, JsonRpcMessage::Response(_)));
    }

    #[test]
    fn buffered_sse_parses_many_events_with_one_event_copy_at_a_time() {
        let mut body = String::new();
        for _ in 0..256 {
            body.push_str("data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n");
        }
        body.push_str("data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n");
        // Leave room for one previous and one current parsed JSON-RPC event.
        let cap = body.len() + 10_000;
        let message = parse_buffered_sse_terminal(body.as_bytes(), Some(cap), &test_signal())
            .unwrap()
            .expect("terminal response");
        assert!(matches!(message, JsonRpcMessage::Response(_)));
        assert!(
            body.len() * 2 > cap,
            "copying every data event would exceed this allowance"
        );
    }

    #[test]
    fn buffered_sse_rejects_string_growth_before_allocating_it() {
        let body = format!("data: {}\ndata: y\n\n", "x".repeat(100_000));
        let cap = body.len() * 3 + 1_000;
        let signal = test_signal();
        assert!(matches!(
            parse_buffered_sse_terminal(body.as_bytes(), Some(cap), &signal),
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit }) if *limit == cap
        ));
    }

    #[test]
    fn parse_buffered_sse_terminal_returns_error_message_as_terminal() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32000,\"message\":\"boom\"}}\n\n";
        let message = parse_buffered_sse_terminal(body.as_bytes(), None, &test_signal())
            .unwrap()
            .expect("terminal error");
        assert!(matches!(message, JsonRpcMessage::Error(_)));
    }

    #[test]
    fn parse_buffered_sse_terminal_without_json_is_none() {
        assert!(
            parse_buffered_sse_terminal(b"data: not-json\n\n", None, &test_signal())
                .unwrap()
                .is_none()
        );
        assert!(
            parse_buffered_sse_terminal(b"", None, &test_signal())
                .unwrap()
                .is_none()
        );
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
    #[expect(
        clippy::too_many_lines,
        reason = "regression checks the wire, parse, and signal boundaries"
    )]
    fn budgeted_tool_json_rejects_numeric_expansion_before_rmcp_parse() {
        let numbers = vec!["1e15"; 300_000].join(",");
        let wire = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"content\":[],\"structuredContent\":{{\"values\":[{numbers}]}}}}}}"
        );
        let parse_cap = 2 * 1_048_576;
        assert!(
            wire.len() < tool_result_wire_cap(262_144),
            "the existing wire limit admits this response"
        );
        assert!(
            !json_preparse_fits(wire.as_bytes(), Some(parse_cap)),
            "dense exponent values must be rejected before rmcp allocates them"
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .unwrap();
        let signal = test_signal();
        let mut response = sub_response(200, Some("application/json"), b"");
        response.body = Bytes::from(wire);
        let result = classify_buffered_post_response(response, &call, false, Some(parse_cap), &signal);
        assert!(
            matches!(
                result,
                Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
            ),
            "the buffered transport must reject before deserialization"
        );
        assert!(matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit }) if *limit == parse_cap));
        assert!(
            json_preparse_fits(
                br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}]}}"#,
                Some(parse_cap)
            ),
            "an ordinary tool result must still fit"
        );
    }

    #[test]
    fn dense_structured_content_is_rejected_before_rmcp_parse() {
        // rmcp's untagged response deserializer can hold intermediate Value
        // trees while constructing the final nested structuredContent value.
        let values = vec![r#"{"a":[0]}"#; 10_000].join(",");
        let wire = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"content\":[],\"structuredContent\":{{\"values\":[{values}]}}}}}}"
        );
        let parse_cap = 6 * 1_048_576;
        assert!(wire.len() < tool_result_wire_cap(3 * 1_048_576));
        assert!(wire.len() < parse_cap, "the raw body fits the parse allowance");
        let signal = test_signal();
        assert!(matches!(
            admit_json_preparse(wire.as_bytes(), Some(parse_cap), &signal),
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit }) if *limit == parse_cap));
    }

    #[test]
    fn oversized_json_rpc_error_keeps_http_status_without_parsing() {
        let numbers = vec!["1e15"; 3_000].join(",");
        let wire = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{{\"code\":-32000,\"message\":\"failure\",\"data\":[{numbers}]}}}}"
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .unwrap();
        let signal = test_signal();
        let mut response = sub_response(500, Some("application/json"), b"");
        response.body = Bytes::from(wire);
        let result = classify_buffered_post_response(response, &call, false, Some(40_000), &signal);
        assert!(
            matches!(result, Err(StreamableHttpError::UnexpectedServerResponse(message)) if message.contains("HTTP 500")),
            "a budgeted error body must fall back to the real HTTP status"
        );
        assert!(signal.get().is_none(), "an HTTP error body must not record a false 413");
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
            1_024,
            Some(4_096),
            None,
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .expect("deserialize tools/call");
        assert_eq!(client.response_limit(&call), tool_result_wire_cap(2048));
        assert_eq!(client.preparse_limit(&call), Some(4_096));

        let list: ClientJsonRpcMessage =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#)
                .expect("deserialize tools/list");
        assert_eq!(client.response_limit(&list), 1_024);
        assert_eq!(
            client.preparse_limit(&list),
            Some(1_024 * crate::mcp_client::MCP_CONTROL_RESPONSE_PEAK_MULTIPLIER)
        );
    }

    #[test]
    fn minimum_tool_result_budget_admits_initialize_control_json() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            1_024,
            1_024,
            Some(2_048),
            None,
        );
        let initialize: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#,
        )
        .expect("deserialize initialize request");
        let wire = br#"{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"test","version":"1.0"}}}"#;
        assert!(json_preparse_peak_bytes(wire).unwrap() > 2_048);
        assert_eq!(client.response_limit(&initialize), 1_024);
        assert_eq!(
            client.preparse_limit(&initialize),
            Some(1_024 * crate::mcp_client::MCP_CONTROL_RESPONSE_PEAK_MULTIPLIER)
        );
        let response = sub_response(200, Some("application/json"), wire);
        assert!(matches!(
            classify_buffered_post_response(
                response,
                &initialize,
                false,
                client.preparse_limit(&initialize),
                &test_signal()
            ),
            Ok(StreamableHttpPostResponse::Json(_, _))
        ));
    }

    #[test]
    fn minimum_tool_result_budget_still_rejects_numeric_result_expansion() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            1_024,
            MAX_CONTROL_RESPONSE_BYTES,
            Some(2_048),
            None,
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .expect("deserialize tools/call");
        let numbers = vec!["1e15"; 200].join(",");
        let wire = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"content\":[],\"structuredContent\":{{\"values\":[{numbers}]}}}}}}"
        );
        assert!(wire.len() < client.response_limit(&call));
        let signal = test_signal();
        let mut response = sub_response(200, Some("application/json"), b"");
        response.body = Bytes::from(wire);
        assert!(matches!(
            classify_buffered_post_response(response, &call, false, client.preparse_limit(&call), &signal),
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 2_048 })
        ));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one numeric control witness covers JSON and SSE admission"
    )]
    fn budgeted_numeric_control_json_and_sse_fit_the_control_reservation() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            1_024,
            1_024,
            Some(2_048),
            None,
        );
        let initialize: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#,
        )
        .expect("deserialize initialize request");
        let numbers = vec!["1e15"; 140].join(",");
        let wire = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":0,\"error\":{{\"code\":-32000,\"message\":\"failure\",\"data\":[{numbers}]}}}}"
        );
        let cap = client.preparse_limit(&initialize).expect("budgeted control cap");
        assert!(wire.len() <= client.response_limit(&initialize));
        let peak = json_preparse_peak_bytes(wire.as_bytes()).unwrap();
        assert!(peak > 2_048 && peak <= cap);

        let sse = format!("data: {wire}\n\n");
        let response = SubResponse {
            body: Bytes::from(wire),
            ..sub_response(200, Some("application/json"), b"")
        };
        assert!(matches!(
            classify_buffered_post_response(response, &initialize, false, Some(cap), &test_signal()),
            Ok(StreamableHttpPostResponse::Json(JsonRpcMessage::Error(_), _))
        ));

        assert!(sse.len() <= client.response_limit(&initialize));
        assert!(matches!(
            parse_buffered_sse_terminal(sse.as_bytes(), Some(cap), &test_signal()),
            Ok(Some(JsonRpcMessage::Error(_)))
        ));
    }

    #[tokio::test]
    async fn budgeted_control_streaming_error_preserves_http_status_when_over_wire_cap() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            1_024,
            1_024,
            Some(2_048),
            None,
        );
        let (body, _cancelled) = fake_body([Bytes::from(vec![b'x'; 1_025])]);
        let signal = test_signal();
        let result = client
            .classify_streaming_post_response(
                sub_response(500, Some("application/json"), b""),
                body,
                false,
                1_024,
                DEFAULT_MAX_SSE_EVENT_SIZE,
                client.control_preparse_limit(),
                Arc::clone(&signal),
            )
            .await;
        assert!(matches!(
            result,
            Err(StreamableHttpError::UnexpectedServerResponse(message)) if message.contains("HTTP 500")
        ));
        assert!(
            signal.get().is_none(),
            "bounded error bodies must not set a false size signal"
        );
    }

    #[test]
    fn control_client_never_exceeds_the_control_ceiling() {
        let client = McpSubrequestClient::control(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            None,
        );
        let call: ClientJsonRpcMessage = serde_json::from_str(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#,
        )
        .expect("deserialize tools/call");
        assert_eq!(client.response_limit(&call), MAX_CONTROL_RESPONSE_BYTES);
    }

    #[test]
    fn mcp_exchange_deadline_is_capped_by_parent_loop_deadline() {
        let now = Instant::now();
        let parent_deadline = now + Duration::from_millis(25);
        let callout = McpCallout::fabricated(false)
            .expect("fabricated callout")
            .with_parent_deadline_for_test(parent_deadline);

        assert_eq!(
            callout.deadline(now, Duration::from_secs(5)),
            parent_deadline,
            "initialize, list, and call exchanges must share the remaining IRR deadline"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the test establishes trusted parent context and inspects one staged MCP exchange"
    )]
    async fn mcp_projects_trace_context_into_every_prepared_exchange() {
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert("x-request-id", HeaderValue::from_static("request-parent"));
        request.headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        let mut context = make_filter_context(&request);
        let filter = TraceContextFilter::from_config(&serde_yaml::from_str("{}").expect("valid config"))
            .expect("trace_context filter");
        let _action = filter
            .on_request(&mut context)
            .await
            .expect("trace context established");
        let parent = context
            .extensions
            .get::<TraceContext>()
            .expect("typed trace context")
            .clone();
        let pipeline = build_bare_outbound_pipeline(true).expect("outbound pipeline");
        let callout = McpCallout::from_context(&context, pipeline).expect("MCP callout context");
        let client = McpSubrequestClient::control(callout, Duration::from_secs(5), None);
        let signal = client.signal_handle();

        let (_executor, _request, extensions, _deadline) = client
            .prepare_staged_request(
                Method::POST,
                "http://127.0.0.1:8321/mcp",
                Bytes::new(),
                HeaderMap::new(),
                MAX_CONTROL_RESPONSE_BYTES,
                MAX_CONTROL_RESPONSE_BYTES,
                &signal,
            )
            .await
            .expect("request prepares without dialing");
        let projected = extensions.get::<TraceContext>().expect("trace context projected");
        assert_eq!(projected.request_id(), parent.request_id());
        assert_eq!(projected.trace_id(), parent.trace_id());
        assert_eq!(projected.flags(), parent.flags());
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
    fn ssrf_validate_always_blocks_metadata_but_allows_other_link_local_with_private_opt_in() {
        assert!(ssrf_validate(&[addr("169.254.169.254:80")], true).is_err());
        assert!(ssrf_validate(&[addr("169.254.1.1:80")], true).is_ok());
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
        // shared Praxis address classification, so they follow the same gate.
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
            None,
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
            MAX_CONTROL_RESPONSE_BYTES,
            None,
            None,
        );
        let expected_wire = tool_result_wire_cap(max_result_bytes);
        assert_eq!(client.wire_cap(), expected_wire);
        assert_eq!(
            client.stream_cumulative_cap(),
            expected_wire + MAX_CONTROL_RESPONSE_BYTES
        );
    }

    #[test]
    fn budgeted_get_bounds_non_streaming_fallback_before_buffering() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(1),
            1_024,
            1_024,
            Some(2_048),
            None,
        );
        assert_eq!(client.buffered_get_response_cap(), 2_048 / 3);
        assert!(client.stream_cumulative_cap() > client.buffered_get_response_cap());
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
            MAX_CONTROL_RESPONSE_BYTES,
            None,
            None,
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
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
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
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
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
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
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
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
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
        let out = classify_buffered_post_response(response, &message, false, None, &test_signal()).unwrap();
        assert!(matches!(out, StreamableHttpPostResponse::Json(_, _)));
    }

    // -- collect_body fail-closed (F4) --

    fn fake_body(
        chunks: impl IntoIterator<Item = Bytes>,
    ) -> (Box<dyn StreamingResponseBody>, Arc<std::sync::atomic::AtomicBool>) {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(chunks, Arc::clone(&cancelled)),
        );
        (body, cancelled)
    }

    fn erroring_body(
        chunks: impl IntoIterator<Item = Bytes>,
    ) -> (Box<dyn StreamingResponseBody>, Arc<std::sync::atomic::AtomicBool>) {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::erroring_after(chunks, Arc::clone(&cancelled)),
        );
        (body, cancelled)
    }

    #[tokio::test]
    async fn collect_body_fails_closed_on_next_chunk_error() {
        // erroring_after yields all chunks, then next_chunk() returns Err — the
        // praxis-withheld-oversize simulation. collect_body must fail closed with
        // a typed 413, record the signal, cancel, and NOT return the partial buffer.
        let (mut body, cancelled) = erroring_body([Bytes::from_static(b"partial")]);
        let signal = Arc::new(OnceLock::new());
        let result = collect_body(&mut body, 1024, None, &signal).await;
        assert!(matches!(
            result,
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(
            matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit: 1024 })),
            "a next_chunk error on the armed drain records a 413 signal at the cap"
        );
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "collect_body must cancel the body on failure"
        );
    }

    #[tokio::test]
    async fn collect_body_preserves_core_chunk_limit_in_size_signal() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut body: Box<dyn StreamingResponseBody> =
            Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::overflowing_after(
                [Bytes::from_static(b"partial")],
                Arc::clone(&cancelled),
                512,
            ));
        let signal = test_signal();
        let result = collect_body(&mut body, 1_024, None, &signal).await;
        assert!(matches!(
            result,
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 512 })
        ));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "a typed core size error cancels the partial body"
        );
    }

    #[tokio::test]
    async fn collect_body_fails_closed_when_buffer_exceeds_cap() {
        // Two 4-byte chunks exceed a 6-byte cap on the second chunk.
        let (mut body, cancelled) = fake_body([Bytes::from_static(b"aaaa"), Bytes::from_static(b"bbbb")]);
        let signal = Arc::new(OnceLock::new());
        let result = collect_body(&mut body, 6, None, &signal).await;
        assert!(matches!(
            result,
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(
            matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit: 6 })),
            "an over-cap buffer records a 413 signal at the local cap"
        );
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "cap breach cancels the body"
        );
    }

    #[tokio::test]
    async fn collect_body_rejects_json_peak_before_buffering_a_large_chunk() {
        let (mut body, cancelled) = fake_body([Bytes::from(vec![b'x'; 3 * 1_048_576])]);
        let signal = Arc::new(OnceLock::new());
        let result = collect_body(&mut body, 6 * 1_048_576, Some(2 * 1_048_576), &signal).await;
        assert!(matches!(
            result,
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 2_097_152 })
        ));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "parse allowance breach cancels the body"
        );
    }

    #[tokio::test]
    async fn budgeted_json_rejects_a_stream_body_without_chunk_admission() {
        let (body, cancelled) = fake_body([Bytes::from_static(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}")]);
        let signal = test_signal();
        let result = client()
            .classify_streaming_post_response(
                sub_response(200, Some("application/json"), b""),
                body,
                false,
                4_096,
                DEFAULT_MAX_SSE_EVENT_SIZE,
                Some(2_048),
                Arc::clone(&signal),
            )
            .await;
        assert!(matches!(
            result,
            Err(StreamableHttpError::Client(McpTransportError::ResponseTooLarge))
        ));
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 2_048 })
        ));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "unsupported body is cancelled before a JSON chunk is pulled"
        );
    }

    #[tokio::test]
    async fn collect_body_returns_bytes_on_clean_eof() {
        let (mut body, cancelled) = fake_body([Bytes::from_static(b"hello "), Bytes::from_static(b"world")]);
        let signal = Arc::new(OnceLock::new());
        let bytes = collect_body(&mut body, 1024, None, &signal)
            .await
            .expect("clean body collects");
        assert_eq!(bytes.as_ref(), b"hello world");
        assert!(signal.get().is_none(), "a clean body records no size signal");
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "collect_body cancels the body after a clean EOF"
        );
    }

    // -- JSON-parse failure for a Request fails closed, never Accepted (F4) --

    #[tokio::test]
    async fn streaming_post_unparseable_json_is_unexpected_server_response_not_accepted() {
        let (body, _cancelled) = fake_body([Bytes::from_static(b"{not json")]);
        let response = sub_response(200, Some("application/json"), b"");
        let result = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
            .await;
        assert!(
            matches!(result, Err(StreamableHttpError::UnexpectedServerResponse(_))),
            "an unparseable streaming JSON Request response must be a typed error, not Ok(Accepted)"
        );
        assert!(
            !matches!(result, Ok(StreamableHttpPostResponse::Accepted)),
            "Accepted is reserved for one-way messages, never a Request"
        );
    }

    #[test]
    fn classify_buffered_post_response_unparseable_json_is_unexpected_server_response() {
        let message: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "x"}
        }))
        .unwrap();
        let response = sub_response(200, Some("application/json"), b"{not json");
        let result = classify_buffered_post_response(response, &message, false, None, &test_signal());
        assert!(
            matches!(result, Err(StreamableHttpError::UnexpectedServerResponse(_))),
            "an unparseable buffered JSON Request response must be a typed error (was Accepted)"
        );
    }

    // -- R4 regression: one-way acks still classify as Accepted --

    #[test]
    fn classify_buffered_post_response_empty_200_notification_is_accepted() {
        // A framing-only success for a one-way notification is an ack, not an error.
        let notification: ClientJsonRpcMessage =
            serde_json::from_value(serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
                .expect("deserialize notification");
        assert!(matches!(notification, ClientJsonRpcMessage::Notification(_)));
        let response = sub_response(200, None, b"");
        let out = classify_buffered_post_response(response, &notification, false, None, &test_signal()).unwrap();
        assert!(
            matches!(out, StreamableHttpPostResponse::Accepted),
            "an empty-200 notification ack must stay Accepted (R4)"
        );
    }

    #[test]
    fn classify_buffered_post_response_202_is_accepted() {
        let message: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "x"}
        }))
        .unwrap();
        let response = sub_response(202, None, b"");
        let out = classify_buffered_post_response(response, &message, false, None, &test_signal()).unwrap();
        assert!(
            matches!(out, StreamableHttpPostResponse::Accepted),
            "a 202 ack must stay Accepted (R4)"
        );
    }

    // -- collect_capped: bounded error-body collect, no size signal (F5) --

    #[tokio::test]
    async fn collect_capped_returns_bytes_within_cap() {
        let (mut body, cancelled) = fake_body([Bytes::from_static(b"err"), Bytes::from_static(b"or")]);
        let out = collect_capped(&mut body, 1024).await;
        assert_eq!(out.as_deref(), Some(b"error".as_ref()));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "a clean error-body collect cancels the body"
        );
    }

    #[tokio::test]
    async fn collect_capped_returns_none_when_over_cap() {
        // Two 4-byte chunks exceed a 6-byte cap; the error body is discarded so the
        // caller falls back to the HTTP status rather than a spurious 413. By
        // construction collect_capped takes no signal handle, so it cannot record one.
        let (mut body, cancelled) = fake_body([Bytes::from_static(b"aaaa"), Bytes::from_static(b"bbbb")]);
        assert!(collect_capped(&mut body, 6).await.is_none());
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "an over-cap error body cancels the body"
        );
    }

    #[tokio::test]
    async fn collect_capped_returns_none_on_transport_error() {
        let (mut body, cancelled) = erroring_body([Bytes::from_static(b"partial")]);
        assert!(collect_capped(&mut body, 1024).await.is_none());
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "a transport error on the error path cancels the body"
        );
    }

    // -- non-2xx streaming arm surfaces a structured JSON-RPC error (F5) --

    #[tokio::test]
    async fn streaming_post_non_2xx_json_rpc_error_is_surfaced_as_json() {
        // The buffered path parses a structured JSON-RPC error out of a non-2xx
        // JSON body; the streaming path must do the same instead of collapsing the
        // reply to a generic "HTTP 500".
        let (body, _cancelled) = fake_body([Bytes::from(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"boom"}}"#.to_owned(),
        )]);
        let response = sub_response(500, Some("application/json"), b"");
        let out = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
            .await
            .expect("a JSON-RPC error is a valid reply, surfaced as Json");
        assert!(matches!(
            out,
            StreamableHttpPostResponse::Json(JsonRpcMessage::Error(_), _)
        ));
    }

    #[tokio::test]
    async fn streaming_post_non_2xx_non_jsonrpc_body_falls_back_to_status() {
        let (body, _cancelled) = fake_body([Bytes::from_static(b"internal error, not json-rpc")]);
        let response = sub_response(500, Some("application/json"), b"");
        let client = client();
        let result = client
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
            .await;
        let Err(StreamableHttpError::UnexpectedServerResponse(msg)) = result else {
            panic!("expected UnexpectedServerResponse for a non-JSON-RPC 500 body");
        };
        assert!(msg.contains("HTTP 500"), "should surface the HTTP status, got {msg}");
        assert!(
            client.signal_handle().get().is_none(),
            "an unparseable error body must never record a 413 size signal"
        );
    }

    #[tokio::test]
    async fn streaming_post_non_2xx_non_json_content_type_falls_back_to_status() {
        let (body, cancelled) = fake_body([Bytes::from_static(b"boom")]);
        let response = sub_response(503, Some("text/plain"), b"");
        let result = client()
            .classify_streaming_post_response(response, body, false, 1024, 16 * 1024 * 1024, None, test_signal())
            .await;
        assert!(matches!(result, Err(StreamableHttpError::UnexpectedServerResponse(_))));
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "a non-JSON error body is drained and cancelled"
        );
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
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024, test_signal().into())
            .await
            .unwrap();
        let mut stream = stream;
        let first = futures::StreamExt::next(&mut stream).await.expect("event").expect("ok");
        assert_eq!(first.data.as_deref(), Some("{\"jsonrpc\":\"2.0\"}"));
    }

    #[tokio::test]
    async fn get_stream_uses_tool_parse_allowance_on_reconnect() {
        let client = McpSubrequestClient::for_tool(
            McpCallout::fabricated(false).expect("fabricated callout"),
            Duration::from_secs(5),
            1_024,
            1_024,
            Some(2_048),
            None,
        );
        let numbers = vec!["1e15"; 140].join(",");
        let wire = format!("data: {{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"values\":[{numbers}]}}}}\n\n");
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([Bytes::from(wire)], cancelled),
        );
        let signal = test_signal();
        let mut stream = client
            .classify_get_stream_response(
                sub_response(200, Some("text/event-stream"), b""),
                Some(body),
                DEFAULT_MAX_SSE_EVENT_SIZE,
                Arc::clone(&signal).into(),
            )
            .await
            .unwrap();
        assert!(futures::StreamExt::next(&mut stream).await.expect("one event").is_err());
        assert!(matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit }) if *limit == 2_048));
    }

    // -- F6: rmcp's max_sse_event_size backstop must never clamp below the wire cap --

    /// rmcp always passes its `config.max_sse_event_size` (16 MiB default) into
    /// the sized-variant overrides, so a client whose wire cap exceeds that
    /// default would have had every event wrongly rejected. The transport must
    /// raise the backstop to the in-play wire tier before handing it to the SSE
    /// adapter. This test drives the pathology directly: `max_sse_event_size = 8`
    /// is far below both the event's retained bytes and the `client()` wire cap
    /// (`tool_result_wire_cap(1024)`), so pre-fix the clamp rejected the event and
    /// post-fix the wire cap wins and the event forwards.
    #[tokio::test]
    async fn get_stream_backstop_never_clamps_below_wire_cap() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> =
            Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
                [Bytes::from_static(b"data: {\"jsonrpc\":\"2.0\"}\n\n")],
                cancelled,
            ));
        let response = sub_response(200, Some("text/event-stream"), b"");
        let mut stream = client()
            .classify_get_stream_response(response, Some(body), 8, test_signal().into())
            .await
            .unwrap();
        let first = futures::StreamExt::next(&mut stream).await.expect("event").expect("ok");
        assert_eq!(first.data.as_deref(), Some("{\"jsonrpc\":\"2.0\"}"));
    }

    /// Streaming-POST twin: a tiny `max_sse_event_size` must not clamp the
    /// per-event bound below the per-message cap the caller passes in.
    #[tokio::test]
    async fn streaming_post_backstop_never_clamps_below_per_message_cap() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body = Box::new(crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks(
            [Bytes::from_static(
                b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n",
            )],
            Arc::clone(&cancelled),
        ));
        let response = sub_response(200, Some("text/event-stream"), b"");
        let out = client()
            .classify_streaming_post_response(response, body, false, 1024, 8, None, test_signal())
            .await
            .unwrap();
        let StreamableHttpPostResponse::Sse(mut stream, _) = out else {
            panic!("expected an SSE post response");
        };
        let first = futures::StreamExt::next(&mut stream).await.expect("event").expect("ok");
        assert_eq!(
            first.data.as_deref(),
            Some("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}")
        );
    }

    #[tokio::test]
    async fn get_stream_405_is_no_sse_support() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([], Arc::clone(&cancelled)),
        );
        let response = sub_response(405, None, b"");
        let result = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024, test_signal().into())
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
            .classify_get_stream_response(response, None, 16 * 1024 * 1024, test_signal().into())
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

    #[test]
    fn get_stream_headers_carry_session_id() {
        let session: Arc<str> = Arc::from("mock-mcp-session-1");
        let headers = client()
            .build_get_stream_headers(None, HashMap::new(), Some(&session), None)
            .unwrap();
        assert_eq!(headers.get(session_id_header()).unwrap(), "mock-mcp-session-1");
        assert_eq!(headers.get(http::header::ACCEPT).unwrap(), "text/event-stream");
    }

    // -- GET stream auth handling (F3) --

    #[tokio::test]
    async fn get_stream_401_with_www_authenticate_is_auth_required() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([], Arc::clone(&cancelled)),
        );
        let mut response = sub_response(401, None, b"");
        response.headers.insert(
            http::header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"mcp\""),
        );
        let result = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024, test_signal().into())
            .await;
        assert!(
            matches!(result, Err(StreamableHttpError::AuthRequired(_))),
            "401 + WWW-Authenticate must surface as AuthRequired"
        );
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "the body must be cancelled on auth failure"
        );
    }

    #[tokio::test]
    async fn get_stream_403_with_www_authenticate_is_insufficient_scope() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([], Arc::clone(&cancelled)),
        );
        let mut response = sub_response(403, None, b"");
        response.headers.insert(
            http::header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer scope=\"admin\""),
        );
        let result = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024, test_signal().into())
            .await;
        assert!(
            matches!(result, Err(StreamableHttpError::InsufficientScope(_))),
            "403 + WWW-Authenticate must surface as InsufficientScope"
        );
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "the body must be cancelled on auth failure"
        );
    }

    #[tokio::test]
    async fn get_stream_401_without_www_authenticate_is_server_does_not_support_sse() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([], Arc::clone(&cancelled)),
        );
        let response = sub_response(401, None, b"");
        let result = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024, test_signal().into())
            .await;
        assert!(
            matches!(result, Err(StreamableHttpError::ServerDoesNotSupportSse)),
            "401 without WWW-Authenticate falls through to ServerDoesNotSupportSse"
        );
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "the body is still cancelled"
        );
    }

    #[tokio::test]
    async fn get_stream_405_remains_server_does_not_support_sse() {
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body: Box<dyn StreamingResponseBody> = Box::new(
            crate::mcp_client::sse_adapter::FakeStreamingBody::from_chunks([], Arc::clone(&cancelled)),
        );
        let response = sub_response(405, None, b"");
        let result = client()
            .classify_get_stream_response(response, Some(body), 16 * 1024 * 1024, test_signal().into())
            .await;
        assert!(
            matches!(result, Err(StreamableHttpError::ServerDoesNotSupportSse)),
            "405 continues to map to ServerDoesNotSupportSse (regression guard)"
        );
        assert!(
            cancelled.load(std::sync::atomic::Ordering::SeqCst),
            "the body is cancelled"
        );
    }
}
