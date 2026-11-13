// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! A2A protocol classifier filter for body-aware routing.

pub(crate) mod config;
pub(crate) mod envelope;
pub(crate) mod sse;
pub(crate) mod task_routing;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::err_expect,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests;

use std::{borrow::Cow, ops::ControlFlow, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::{is_event_stream_content_type, promotion::contains_control_chars};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    builtins::http::payload_processing::{MAX_DYNAMIC_VALUE_LEN, OnInvalidBehavior, json_rpc::config::JsonRpcConfig},
    parse_filter_config,
};
use tracing::{debug, trace};

use self::{
    config::{A2aConfig, build_config},
    envelope::{A2aEnvelope, extract_a2a_envelope},
    task_routing::{LocalTaskRouteStore, RouteSource, attempt_route_lookup},
};

// -----------------------------------------------------------------------------
// A2aFilter
// -----------------------------------------------------------------------------

/// Extracts A2A protocol metadata from JSON-RPC request bodies and promotes
/// method, family, task ID, streaming detection, and version to request headers,
/// filter results, and durable metadata for routing.
///
/// When `task_routing.enabled` is true, the filter captures task and context
/// ownership from backend responses and uses it to route follow-up requests.
/// Task-owner routing sends `GetTask`, `CancelTask`, `SubscribeToTask`, and
/// push-notification config methods back to the backend that created the task.
/// Context-owner routing sends `ListTasks`, `SendMessage`, and
/// `SendStreamingMessage` requests carrying a known `contextId` back to the
/// backend that owns the context. Task-ID routes take precedence over
/// context-ID routes. Context routes always use `ttl_seconds`; a completed
/// task does not evict the context route.
///
/// # YAML
///
/// ```yaml
/// filter: a2a
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: a2a
/// max_body_bytes: 65536
/// on_invalid: reject
/// method_aliases:
///   message/send: SendMessage
///   message/stream: SendStreamingMessage
///   tasks/get: GetTask
///   tasks/cancel: CancelTask
/// headers:
///   method: x-praxis-a2a-method
///   family: x-praxis-a2a-family
///   context_id: x-praxis-a2a-context-id
///   task_id: x-praxis-a2a-task-id
///   kind: x-praxis-a2a-kind
///   streaming: x-praxis-a2a-streaming
///   version: x-praxis-a2a-version
/// task_routing:
///   enabled: true
///   store: local
///   route_cluster_header: x-praxis-a2a-route-cluster
///   ttl_seconds: 3600
///   terminal_ttl_seconds: 300
///   max_response_body_bytes: 65536
/// ```
pub struct A2aFilter {
    /// Parsed filter configuration.
    config: A2aConfig,

    /// Shared JSON-RPC parser configuration.
    json_rpc_config: JsonRpcConfig,

    /// Maximum body bytes for `StreamBuffer`.
    max_body_bytes: usize,

    /// Local task route store, present only when task routing is enabled.
    task_route_store: Option<Arc<LocalTaskRouteStore>>,
}

impl A2aFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: A2aConfig = parse_filter_config("a2a", config)?;
        let validated_config = build_config(cfg)?;
        let max_body_bytes = validated_config.max_body_bytes;
        let json_rpc_config = build_json_rpc_config(max_body_bytes);

        let task_route_store = (validated_config.task_routing.enabled
            && matches!(validated_config.task_routing.store, Some(config::TaskRouteStore::Local)))
        .then(|| Arc::new(LocalTaskRouteStore::new()));

        Ok(Box::new(Self {
            config: validated_config,
            json_rpc_config,
            max_body_bytes,
            task_route_store,
        }))
    }
}

#[async_trait]
impl HttpFilter for A2aFilter {
    fn name(&self) -> &'static str {
        "a2a"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        if self.task_route_store.is_some() {
            BodyAccess::ReadOnly
        } else {
            BodyAccess::None
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    #[expect(clippy::too_many_lines, reason = "31 lines; one over limit due to trace! fields")]
    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let parsed = match praxis_filter::builtins::http::payload_processing::body_parsing::parse_json_rpc_body(
            body.as_ref(),
            end_of_stream,
            &self.json_rpc_config,
            self.config.on_invalid,
        ) {
            Ok(Some(p)) => p,
            Ok(None) => return Ok(FilterAction::Continue),
            Err(action) => return action,
        };

        let a2a_envelope = extract_a2a_envelope(
            &parsed.value,
            &parsed.method,
            &self.config.method_aliases,
            &ctx.request.headers,
        );

        write_metadata(ctx, &parsed.envelope, &a2a_envelope);
        promote_a2a_headers(
            &a2a_envelope,
            &parsed.envelope,
            &self.config,
            &mut ctx.extra_request_headers,
        );
        promote_filter_results(ctx, &parsed.envelope, &a2a_envelope)?;

        if let Some(store) = &self.task_route_store {
            lookup_task_route(ctx, &a2a_envelope, store, &self.config);
        }

        trace!(
            a2a_method = a2a_envelope.method.as_str(),
            a2a_family = a2a_envelope.family.as_str(),
            streaming = a2a_envelope.streaming,
            context_id = ?a2a_envelope.context_id,
            task_id = ?a2a_envelope.task_id,
            version = ?a2a_envelope.version,
            "extracted A2A envelope metadata"
        );

        Ok(FilterAction::Release)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if self.task_route_store.is_none() {
            return Ok(FilterAction::Continue);
        }

        let method = ctx.get_metadata("a2a.method");
        let is_send_message = method.is_some_and(|m| m == "SendMessage");
        let is_sse_capable = method.is_some_and(|m| m == "SendStreamingMessage" || m == "SubscribeToTask");

        if !is_send_message && !is_sse_capable {
            return Ok(FilterAction::Continue);
        }

        let is_success = ctx.response_header.as_ref().is_some_and(|r| r.status.is_success());

        if !is_success {
            return Ok(FilterAction::Continue);
        }

        let is_sse = ctx
            .response_header
            .as_ref()
            .and_then(|r| r.headers.get("content-type"))
            .and_then(|v| v.to_str().ok())
            .is_some_and(is_event_stream_content_type);

        if is_sse_capable && is_sse {
            start_capture(ctx, CapturedBody::Sse(sse::SseScanState::default()));
            return Ok(FilterAction::Continue);
        }

        if is_sse {
            ctx.filter_metadata
                .insert("a2a.response.is_sse".to_owned(), "true".to_owned());
            return Ok(FilterAction::Continue);
        }

        start_capture(ctx, CapturedBody::Json(JsonCapture::default()));

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let Some(store) = &self.task_route_store else {
            return Ok(FilterAction::Continue);
        };
        let Some(capture) = ctx.get_filter_state_mut::<ResponseCapture>() else {
            return Ok(FilterAction::Continue);
        };

        if capture
            .advance(body.as_deref(), end_of_stream, store, &self.config.task_routing)
            .is_break()
        {
            ctx.remove_filter_state::<ResponseCapture>();
        }

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// ResponseCapture
// -----------------------------------------------------------------------------

/// Task-route capture of one backend response, kept in `filter_state`
/// from `on_response` until its routes are stored or capture is abandoned.
struct ResponseCapture {
    /// Cluster that served the response; captured routes point at it.
    cluster: String,

    /// Capture progress through the response body.
    body: CapturedBody,
}

/// Capture progress through a response body, by framing.
enum CapturedBody {
    /// A non-streaming JSON response, buffered until it balances or ends.
    Json(JsonCapture),

    /// A JSON response that balanced and parsed before end of stream. Its
    /// routes are stored only if nothing but whitespace follows it.
    Tentative(serde_json::Value),

    /// An SSE response, scanned for task events as it streams.
    Sse(sse::SseScanState),
}

/// A non-streaming JSON response being buffered for task-route capture.
#[derive(Default)]
struct JsonCapture {
    /// Response bytes so far, at most `max_response_body_bytes`.
    buffer: Vec<u8>,

    /// Bracket balance of `buffer`, updated chunk by chunk.
    balance: JsonBalanceState,
}

/// Starts task-route capture of this response, if the cluster that served
/// it is known.
fn start_capture(ctx: &mut HttpFilterContext<'_>, body: CapturedBody) {
    if let Some(cluster) = ctx.cluster_name().map(str::to_owned) {
        ctx.insert_filter_state(ResponseCapture { cluster, body });
    }
}

impl ResponseCapture {
    /// Feeds one `on_response_body` call to the capture and stores the
    /// routes it completes. Breaks once the rest of the body cannot change
    /// the outcome, so the caller can drop the capture.
    #[expect(clippy::too_many_lines, reason = "one arm per capture mode")]
    fn advance(
        &mut self,
        chunk: Option<&[u8]>,
        end_of_stream: bool,
        store: &LocalTaskRouteStore,
        config: &config::TaskRoutingConfig,
    ) -> ControlFlow<()> {
        match &mut self.body {
            CapturedBody::Json(json) => {
                let Some(response) = json.feed(chunk, end_of_stream, config.max_response_body_bytes)? else {
                    return ControlFlow::Continue(());
                };
                if end_of_stream {
                    store_task_route(&response, &self.cluster, store, config);
                    return ControlFlow::Break(());
                }
                // Structurally complete but EOS not yet confirmed. Hold the
                // parsed response: the route is committed only when EOS
                // arrives with no further non-whitespace bytes. This prevents
                // a valid JSON prefix followed by trailing garbage from
                // poisoning task-route ownership.
                self.body = CapturedBody::Tentative(response);
                ControlFlow::Continue(())
            },
            CapturedBody::Tentative(response) => {
                let has_new_content =
                    chunk.is_some_and(|b| b.iter().any(|&byte| !matches!(byte, b' ' | b'\t' | b'\n' | b'\r')));
                if has_new_content {
                    debug!("tentative route discarded: trailing non-whitespace after balanced JSON prefix");
                    return ControlFlow::Break(());
                }
                if !end_of_stream {
                    // Whitespace-only chunk, EOS not yet seen: keep waiting.
                    return ControlFlow::Continue(());
                }
                debug!("tentative route confirmed at EOS: no trailing content");
                store_task_route(response, &self.cluster, store, config);
                ControlFlow::Break(())
            },
            CapturedBody::Sse(scanner) => {
                if let Some(chunk) = chunk {
                    process_sse_response_chunk(scanner, &self.cluster, chunk, store, config);
                }
                if !end_of_stream {
                    return ControlFlow::Continue(());
                }
                flush_pending_sse_payloads(scanner, &self.cluster, store, config);
                ControlFlow::Break(())
            },
        }
    }
}

impl JsonCapture {
    /// Appends `chunk` to the buffer and parses the buffer once its
    /// brackets balance or the stream ends. Continues with `None` while
    /// more bytes are needed. Breaks when capture must be abandoned: the
    /// body outgrew `max_bytes`, its brackets can no longer balance, or it
    /// failed to parse.
    fn feed(
        &mut self,
        chunk: Option<&[u8]>,
        end_of_stream: bool,
        max_bytes: usize,
    ) -> ControlFlow<(), Option<serde_json::Value>> {
        if let Some(chunk) = chunk {
            if self.buffer.len().saturating_add(chunk.len()) > max_bytes {
                debug!(
                    existing_bytes = self.buffer.len(),
                    chunk_len = chunk.len(),
                    max_bytes,
                    "response body exceeds capture limit, skipping route capture"
                );
                return ControlFlow::Break(());
            }
            self.buffer.extend_from_slice(chunk);
            scan_json_balance(&mut self.balance, chunk);
        }

        if self.balance.invalid {
            // Mismatched or unmatched brackets are permanent: no further
            // bytes can make this buffer type-matched. Abandon capture now
            // instead of buffering every remaining chunk up to
            // max_response_body_bytes only to hit this same dead end at
            // end_of_stream.
            return ControlFlow::Break(());
        }
        if !self.balance.is_complete() && !end_of_stream {
            return ControlFlow::Continue(None);
        }

        // A failed parse is terminal: later bytes cannot repair a balanced
        // but invalid buffer (e.g. a trailing comma), and at end of stream
        // there are none.
        match serde_json::from_slice(&self.buffer) {
            Ok(response) => ControlFlow::Continue(Some(response)),
            Err(_) => ControlFlow::Break(()),
        }
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Look up a task or context route and inject the route cluster header on hit.
///
/// Task routes take precedence over context routes. For task-routable methods,
/// only the task ID is consulted. For context-routable methods, the context ID
/// is consulted. Because the method sets are disjoint in the current A2A spec,
/// both lookups cannot apply in practice - but the ordering guarantees task
/// wins if both IDs are ever simultaneously present.
#[expect(clippy::too_many_lines, reason = "sequential lookup-classify-trace pipeline")]
fn lookup_task_route(
    ctx: &mut HttpFilterContext<'_>,
    a2a_envelope: &A2aEnvelope,
    store: &LocalTaskRouteStore,
    config: &A2aConfig,
) {
    let task_id = if a2a_envelope.method.is_task_routable() {
        a2a_envelope.task_id.as_deref()
    } else {
        None
    };

    let context_id = if a2a_envelope.method.is_context_routable() {
        a2a_envelope.context_id.as_deref()
    } else {
        None
    };

    if task_id.is_none() && context_id.is_none() {
        return;
    }

    let has_task_id = task_id.is_some();
    let has_context_id = context_id.is_some();

    if let Some((cluster, source)) = attempt_route_lookup(store, task_id, context_id) {
        ctx.extra_request_headers.push((
            Cow::Owned(config.task_routing.route_cluster_header.clone()),
            (*cluster).to_owned(),
        ));

        let decision = match source {
            RouteSource::Task => "task_route_hit",
            RouteSource::Context => "context_route_hit",
        };
        ctx.set_metadata("a2a.route_decision", decision);
        ctx.set_metadata("a2a.route_source", source.as_str());
        ctx.set_metadata("a2a.route_cluster", &*cluster);

        debug!(
            has_task_id,
            task_id_len = task_id.map_or(0, str::len),
            has_context_id,
            context_id_len = context_id.map_or(0, str::len),
            lookup_hit = true,
            route_source = source.as_str(),
            cluster = %cluster,
            method = a2a_envelope.method.as_str(),
            "route lookup hit"
        );
    } else {
        let miss = if has_task_id {
            "task_route_miss"
        } else {
            "context_route_miss"
        };
        ctx.set_metadata("a2a.route_decision", miss);
        debug!(
            has_task_id,
            task_id_len = task_id.map_or(0, str::len),
            has_context_id,
            context_id_len = context_id.map_or(0, str::len),
            lookup_hit = false,
            method = a2a_envelope.method.as_str(),
            "route lookup miss"
        );
    }
}

/// Store task and context routes extracted from a response body.
///
/// Task routes use `terminal_ttl_seconds` when the task is done; context routes
/// always use `ttl_seconds` because a completed task does not end the context -
/// later messages or `ListTasks` calls in the same context still need routing.
#[expect(clippy::too_many_lines, reason = "sequential extract-store-log pipeline")]
fn store_task_route(
    value: &serde_json::Value,
    cluster: &str,
    store: &LocalTaskRouteStore,
    config: &config::TaskRoutingConfig,
) {
    let Some(extracted) = task_routing::extract_task_route(value) else {
        return;
    };

    let task_ttl = task_routing::route_ttl(extracted.terminal, config);

    if extracted.terminal && config.terminal_ttl_seconds == 0 {
        store.remove(&extracted.task_id);
        debug!(
            has_task_id = true,
            task_id_len = extracted.task_id.len(),
            cluster = %cluster,
            "terminal task route removed (terminal_ttl_seconds=0)"
        );
    } else {
        store.put(&extracted.task_id, cluster, task_ttl);
        debug!(
            has_task_id = true,
            task_id_len = extracted.task_id.len(),
            cluster = %cluster,
            terminal = extracted.terminal,
            "stored task route from response"
        );
    }

    // Context routes always use the normal TTL. A completed task does not
    // signal context completion; the same context may receive further messages
    // or ListTasks queries.
    if let Some(ctx_id) = &extracted.context_id {
        let ctx_ttl = std::time::Duration::from_secs(config.ttl_seconds);
        store.put_context(ctx_id, cluster, ctx_ttl);
        debug!(
            has_context_id = true,
            context_id_len = ctx_id.len(),
            cluster = %cluster,
            "stored context route from response"
        );
    }
}

/// Incremental brace/bracket state for detecting when an accumulated
/// buffer holds a complete, type-matched top-level JSON value, without
/// re-parsing the whole buffer on every chunk.
///
/// A2A JSON-RPC responses are always a top-level `{...}` object, so
/// tracking a stack of expected closing brackets while skipping bytes
/// inside string literals is sufficient to detect completion - a full
/// incremental JSON parser is unnecessary. Structural JSON bytes (`{`,
/// `}`, `[`, `]`, `"`, `\`) are always single-byte ASCII, so scanning raw
/// bytes is safe even when a chunk boundary splits a multibyte UTF-8
/// code point.
///
/// This is a heuristic gate, not a validator: a balanced, type-matched
/// top-level value does not guarantee the buffer is valid JSON (e.g.
/// trailing bytes after a complete object, or a trailing comma). The
/// real `serde_json` parse in [`JsonCapture::feed`] remains the source
/// of truth, and it abandons capture immediately on a failed parse - see
/// the comment there for why that is always safe to do unconditionally,
/// without waiting for `end_of_stream`.
#[derive(Debug, Default, Clone)]
#[expect(clippy::struct_excessive_bools, reason = "independent per-byte scan flags")]
struct JsonBalanceState {
    /// Expected closing bytes (`}` for `{`, `]` for `[`) for currently
    /// open brackets, innermost last. `{]` must not read as balanced,
    /// so closers are matched against the opener type, not just counted.
    stack: Vec<u8>,
    /// Whether an opening `{`/`[` has been seen. An empty stack before
    /// this is true means "no object seen yet", not "complete".
    started: bool,
    /// Whether the scan is currently positioned inside a string literal.
    in_string: bool,
    /// Whether the previous byte inside a string was an unconsumed `\`.
    escaped: bool,
    /// Set once a closing bracket does not match the type of its
    /// corresponding opener, or appears with no opener on the stack.
    /// Once set, scanning stops and `is_complete` always reports
    /// `false`, so the caller falls back to `end_of_stream` instead of
    /// treating a structurally invalid buffer as a completed value.
    invalid: bool,
}

impl JsonBalanceState {
    /// Whether the bytes scanned so far form a balanced, type-matched
    /// top-level value. See the struct-level doc comment: this does not
    /// guarantee the buffer is valid JSON.
    fn is_complete(&self) -> bool {
        self.started && !self.invalid && self.stack.is_empty()
    }

    /// Whether further scanning could still change the outcome. `false`
    /// once the buffer is either complete or marked invalid.
    fn is_done(&self) -> bool {
        self.is_complete() || self.invalid
    }
}

/// Update `state` with the bytes in `chunk`. A no-op once `state` is
/// already complete or invalid, so total work across all chunks is
/// O(n) in bytes rather than the O(n^2) of re-scanning the whole buffer
/// every call.
#[expect(
    clippy::too_many_lines,
    reason = "single per-byte match pipeline, now with bracket-type matching"
)]
fn scan_json_balance(state: &mut JsonBalanceState, chunk: &[u8]) {
    if state.is_done() {
        return;
    }

    for &byte in chunk {
        if state.escaped {
            state.escaped = false;
            continue;
        }

        if state.in_string {
            match byte {
                b'\\' => state.escaped = true,
                b'"' => state.in_string = false,
                _ => {},
            }
            continue;
        }

        match byte {
            b'"' => state.in_string = true,
            b'{' => {
                state.started = true;
                state.stack.push(b'}');
            },
            b'[' => {
                state.started = true;
                state.stack.push(b']');
            },
            b'}' | b']' => match state.stack.pop() {
                Some(expected) if expected == byte => {
                    if state.is_complete() {
                        return;
                    }
                },
                _ => {
                    state.invalid = true;
                    return;
                },
            },
            _ => {},
        }
    }
}

/// Runs inside the synchronous `on_response_body` hook, so it cannot
/// await or write to external stores - scanner state persists in
/// `filter_state` between calls.
fn process_sse_response_chunk(
    scanner: &mut sse::SseScanState,
    cluster: &str,
    chunk: &[u8],
    store: &LocalTaskRouteStore,
    config: &config::TaskRoutingConfig,
) {
    let result = sse::scan_sse_chunk(scanner, chunk, config.max_response_body_bytes);

    for payload in &result.payloads {
        try_extract_task_from_sse_payload(payload, cluster, store, config);
    }

    if result.dropped_events > 0 {
        debug!(
            dropped_events = result.dropped_events,
            max_bytes = config.max_response_body_bytes,
            "SSE event exceeded capture limit, discarding and resuming at next event boundary"
        );
    }
}

/// Drains any incomplete SSE event left in the scanner at end-of-stream
/// so the last event is not silently dropped when the provider omits the
/// trailing blank line before closing the connection.
fn flush_pending_sse_payloads(
    scanner: &mut sse::SseScanState,
    cluster: &str,
    store: &LocalTaskRouteStore,
    config: &config::TaskRoutingConfig,
) {
    let mut payloads = Vec::new();
    sse::flush_sse_state(scanner, &mut payloads);

    for payload in &payloads {
        try_extract_task_from_sse_payload(payload, cluster, store, config);
    }
}

/// Invalid UTF-8 or unparseable JSON silently skips - the proxy must
/// never fail on arbitrary SSE payloads.
fn try_extract_task_from_sse_payload(
    data: &[u8],
    cluster: &str,
    store: &LocalTaskRouteStore,
    config: &config::TaskRoutingConfig,
) {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };

    store_task_route(&value, cluster, store, config);
}

/// Build a `JsonRpcConfig` for the shared parser with A2A-appropriate defaults.
fn build_json_rpc_config(max_body_bytes: usize) -> JsonRpcConfig {
    use praxis_filter::builtins::http::payload_processing::json_rpc::config::{BatchPolicy, JsonRpcHeaders};

    let headers = JsonRpcHeaders {
        id: None,
        kind: None,
        method: None,
    };

    JsonRpcConfig {
        batch_policy: BatchPolicy::Reject,
        headers,
        max_batch_size: 100,
        max_body_bytes,
        on_invalid: OnInvalidBehavior::Continue,
    }
}

/// Write durable metadata that persists across all Pingora lifecycle phases.
fn write_metadata(
    ctx: &mut HttpFilterContext<'_>,
    envelope: &praxis_filter::builtins::http::payload_processing::json_rpc::envelope::JsonRpcEnvelope,
    a2a: &A2aEnvelope,
) {
    let method_str = a2a.method.as_str();

    set_safe_metadata(ctx, "json_rpc.method", envelope.method.as_deref());

    if is_promotable(method_str) {
        ctx.set_metadata("a2a.method", method_str);
    }
    ctx.set_metadata("json_rpc.kind", envelope.kind.as_str());

    set_safe_metadata(ctx, "a2a.original_method", a2a.original_method.as_deref());
    ctx.set_metadata("a2a.family", a2a.family.as_str());
    ctx.set_metadata("a2a.streaming", if a2a.streaming { "true" } else { "false" });
    set_safe_metadata(ctx, "a2a.context_id", a2a.context_id.as_deref());
    set_safe_metadata(ctx, "a2a.task_id", a2a.task_id.as_deref());
    set_safe_metadata(ctx, "a2a.version", a2a.version.as_deref());
}

/// Write a dynamic value to durable metadata if it is within promotion bounds.
fn set_safe_metadata(ctx: &mut HttpFilterContext<'_>, key: &str, value: Option<&str>) {
    if let Some(v) = value
        && is_promotable(v)
    {
        ctx.set_metadata(key, v);
    }
}

/// Whether a dynamic value is safe and bounded for promotion to headers/metadata.
fn is_promotable(value: &str) -> bool {
    !contains_control_chars(value) && value.len() <= MAX_DYNAMIC_VALUE_LEN
}

/// Promote A2A metadata to internal request headers.
fn promote_a2a_headers(
    a2a: &A2aEnvelope,
    envelope: &praxis_filter::builtins::http::payload_processing::json_rpc::envelope::JsonRpcEnvelope,
    config: &A2aConfig,
    headers: &mut Vec<(Cow<'static, str>, String)>,
) {
    if let Some(header_name) = &config.headers.method {
        let method_str = a2a.method.as_str();
        if !contains_control_chars(method_str) && method_str.len() <= MAX_DYNAMIC_VALUE_LEN {
            headers.push((Cow::Owned(header_name.clone()), method_str.to_owned()));
        }
    }

    if let Some(header_name) = &config.headers.family {
        headers.push((Cow::Owned(header_name.clone()), a2a.family.as_str().to_owned()));
    }

    promote_optional_header(&config.headers.context_id, a2a.context_id.as_deref(), headers);
    promote_optional_header(&config.headers.task_id, a2a.task_id.as_deref(), headers);

    if let Some(header_name) = &config.headers.kind {
        headers.push((Cow::Owned(header_name.clone()), envelope.kind.as_str().to_owned()));
    }

    if let Some(header_name) = &config.headers.streaming {
        let streaming = if a2a.streaming { "true" } else { "false" };
        headers.push((Cow::Owned(header_name.clone()), streaming.to_owned()));
    }

    promote_optional_header(&config.headers.version, a2a.version.as_deref(), headers);
}

/// Promote a dynamic optional value to a request header if configured and safe.
fn promote_optional_header(
    header_name: &Option<String>,
    value: Option<&str>,
    headers: &mut Vec<(Cow<'static, str>, String)>,
) {
    if let Some(header_name) = header_name
        && let Some(value) = value
        && is_promotable(value)
    {
        headers.push((Cow::Owned(header_name.clone()), value.to_owned()));
    }
}

/// Promote A2A metadata to filter results for router branch conditions.
fn promote_filter_results(
    ctx: &mut HttpFilterContext<'_>,
    envelope: &praxis_filter::builtins::http::payload_processing::json_rpc::envelope::JsonRpcEnvelope,
    a2a: &A2aEnvelope,
) -> Result<(), FilterError> {
    let results = ctx.filter_results.entry("a2a").or_default();

    let method_str = a2a.method.as_str();
    if is_promotable(method_str) {
        results.set("method", method_str.to_owned())?;
    }

    results.set("family", a2a.family.as_str())?;
    results.set("streaming", if a2a.streaming { "true" } else { "false" })?;
    results.set("kind", envelope.kind.as_str())?;

    set_optional_result(results, "context_id", a2a.context_id.as_deref())?;
    set_optional_result(results, "task_id", a2a.task_id.as_deref())?;
    set_optional_result(results, "version", a2a.version.as_deref())?;

    Ok(())
}

/// Set a dynamic optional value in filter results if safe and bounded.
fn set_optional_result(
    results: &mut praxis_filter::FilterResultSet,
    key: &'static str,
    value: Option<&str>,
) -> Result<(), FilterError> {
    if let Some(v) = value
        && is_promotable(v)
    {
        results.set(key, v.to_owned())?;
    }
    Ok(())
}
