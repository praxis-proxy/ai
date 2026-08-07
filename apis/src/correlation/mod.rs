// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Request correlation shared by forwarded and delegated calls.
//!
//! A single AI request reaches the backend over two different
//! paths. The **forwarded** request is the client's own request,
//! proxied upstream. A **delegated** call is one the proxy
//! originates itself while handling that request — a Files API
//! metadata fetch, a vector-store search. Without shared
//! correlation the two appear at the backend as unrelated traffic,
//! so a slow or failing request cannot be attributed to a layer.
//!
//! [`TraceContext`] resolves the identifiers once per downstream
//! request. Every hop then draws its own span from that context,
//! so all legs share a trace-id while remaining individually
//! measurable.
//!
//! # Resolution
//!
//! The request ID follows the same precedence the `request_id`
//! core builtin uses when echoing an ID onto the response:
//!
//! 1. A client-supplied header on the downstream request.
//! 2. An ID injected into `extra_request_headers` by an earlier filter.
//! 3. A freshly generated ID.
//!
//! Step 2 matters: filters that inject headers write to
//! `extra_request_headers` (a pending mutation applied to the
//! upstream request) rather than back into `ctx.request.headers`,
//! so reading the downstream headers alone finds an injected value
//! only when the client happened to supply one.
//!
//! # Agreement across hops
//!
//! Header inspection alone is not enough to make every hop of one
//! request agree. Delegated callouts can run in the `StreamBuffer`
//! pre-read phase, ahead of header-phase filters, and see only the
//! original downstream headers — never a value a filter injected.
//!
//! The resolved [`TraceContext`] is therefore shared through request
//! extensions: whichever hop asks first initializes it, and every
//! later hop reuses it regardless of filter order.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use http::{HeaderMap, HeaderName, HeaderValue};
use praxis_filter::HttpFilterContext;
use tracing::trace;

/// Header carrying the request correlation ID.
const REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Header carrying W3C trace context.
const TRACEPARENT: HeaderName = HeaderName::from_static("traceparent");

/// Trace-context version this proxy emits.
const VERSION: &str = "00";

/// Trace-flags emitted when starting a new trace: sampled.
const SAMPLED: &str = "01";

/// Hex length of a W3C trace-id.
const TRACE_ID_LEN: usize = 32;

/// Hex length of a W3C span-id.
const SPAN_ID_LEN: usize = 16;

/// Correlation identifiers for one downstream request.
///
/// Resolved once, then used to stamp every hop the request makes.
/// Each hop gets its own span-id under the shared trace-id.
///
/// Shared through [`RequestExtensions`] rather than through header
/// mutations, because filter ordering does not guarantee who resolves
/// first. Delegated callouts made from the `StreamBuffer` pre-read
/// phase run before header-phase filters and see only the original
/// downstream headers, so a value injected by a filter would not
/// reach them. Whichever hop resolves first initializes the shared
/// context; every later hop reuses it.
///
/// [`RequestExtensions`]: praxis_filter::RequestExtensions
#[derive(Clone)]
pub struct TraceContext {
    /// Value for the `x-request-id` header.
    request_id: String,
    /// 32 lowercase hex characters shared by every hop.
    trace_id: String,
    /// 2 lowercase hex characters of W3C trace-flags.
    flags: String,
}

impl TraceContext {
    /// Return the request's shared trace context, resolving and
    /// storing it if this is the first hop to ask.
    ///
    /// Prefer this wherever the context is available mutably: it is
    /// what makes every hop of one request agree on a trace-id.
    pub fn get_or_init(ctx: &mut HttpFilterContext<'_>) -> Self {
        if let Some(shared) = ctx.extensions.get::<Self>() {
            return shared.clone();
        }

        let resolved = Self::resolve(ctx);
        ctx.extensions.insert(resolved.clone());
        resolved
    }

    /// Read the request's shared trace context.
    ///
    /// Falls back to resolving a fresh context when no hop has
    /// initialized one yet, so correlation still works in a chain
    /// with no `trace_context` filter.
    pub(crate) fn from_filter_context(ctx: &HttpFilterContext<'_>) -> Self {
        ctx.extensions
            .get::<Self>()
            .cloned()
            .unwrap_or_else(|| Self::resolve(ctx))
    }

    /// Resolve identifiers from the request, ignoring any shared
    /// context.
    fn resolve(ctx: &HttpFilterContext<'_>) -> Self {
        let request_id = resolve_request_id(ctx);
        let (trace_id, flags) = resolve_trace(ctx);

        trace!(%request_id, %trace_id, "resolved correlation for request");

        Self {
            request_id,
            trace_id,
            flags,
        }
    }

    /// The resolved request correlation ID.
    #[must_use]
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Build a `traceparent` for one hop, with a fresh span-id.
    ///
    /// Called once per outbound leg. The forwarded request and each
    /// delegated call are separate spans of the same trace, which is
    /// what makes their latencies separable.
    fn traceparent_for_hop(&self, ctx: &HttpFilterContext<'_>) -> String {
        let span_id = generate_span_id(ctx);
        let Self { trace_id, flags, .. } = self;
        format!("{VERSION}-{trace_id}-{span_id}-{flags}")
    }

    /// Build the correlation headers for one hop.
    #[must_use]
    pub fn headers_for_hop(&self, ctx: &HttpFilterContext<'_>) -> [(HeaderName, String); 2] {
        [
            (REQUEST_ID, self.request_id.clone()),
            (TRACEPARENT, self.traceparent_for_hop(ctx)),
        ]
    }
}

/// Correlation headers for the delegated calls of one request.
///
/// Unlike an operator-configured header allowlist, these are
/// injected unconditionally. Correlation that depends on per-filter
/// configuration is correlation that silently goes missing.
pub(crate) struct Correlation {
    /// Value for the `x-request-id` header.
    request_id: HeaderValue,
    /// Value for the `traceparent` header.
    traceparent: HeaderValue,
}

impl Correlation {
    /// Resolve correlation headers for this request's delegated calls.
    pub(crate) fn from_filter_context(ctx: &HttpFilterContext<'_>) -> Self {
        let context = TraceContext::from_filter_context(ctx);

        // A generated ID is hex; a client-supplied one already passed
        // header parsing to reach us. Fall back rather than fail the
        // callout on an unrepresentable value.
        let to_value = |raw: &str| HeaderValue::from_str(raw).unwrap_or_else(|_| HeaderValue::from_static(""));

        Self {
            request_id: to_value(context.request_id()),
            traceparent: to_value(&context.traceparent_for_hop(ctx)),
        }
    }

    /// Insert correlation headers into an outbound callout map.
    ///
    /// Inserts rather than appends, so correlation overwrites any
    /// same-named header copied from the downstream request.
    pub(crate) fn apply(&self, map: &mut HeaderMap) {
        map.insert(REQUEST_ID, self.request_id.clone());
        map.insert(TRACEPARENT, self.traceparent.clone());
    }
}

/// Resolve the request ID, generating one if nothing upstream
/// supplied or injected it.
fn resolve_request_id(ctx: &HttpFilterContext<'_>) -> String {
    if let Some(client_id) = ctx.request.headers.get(&REQUEST_ID).and_then(|v| v.to_str().ok()) {
        return client_id.to_owned();
    }

    if let Some(injected) = injected_header(ctx, REQUEST_ID.as_str()) {
        return injected;
    }

    ctx.id_generator.generate(ctx.time_source)
}

/// Resolve the trace-id and flags shared by every hop.
///
/// Continues a valid inbound or injected trace; otherwise starts a
/// new sampled one. A value that fails validation is discarded
/// rather than carried forward — it is client-controlled input that
/// would otherwise reach the telemetry backend unchecked.
fn resolve_trace(ctx: &HttpFilterContext<'_>) -> (String, String) {
    let inbound = ctx
        .request
        .headers
        .get(&TRACEPARENT)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent)
        .or_else(|| {
            injected_header(ctx, TRACEPARENT.as_str())
                .as_deref()
                .and_then(parse_traceparent)
        });

    if let Some(InboundTrace { trace_id, flags }) = inbound {
        return (trace_id, flags);
    }

    (generate_trace_id(ctx), SAMPLED.to_owned())
}

/// Look up a header injected by an earlier filter into
/// `extra_request_headers`.
fn injected_header(ctx: &HttpFilterContext<'_>, name: &str) -> Option<String> {
    ctx.extra_request_headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

/// Trace-id and flags carried forward from a valid `traceparent`.
struct InboundTrace {
    /// 32 lowercase hex characters.
    trace_id: String,
    /// 2 lowercase hex characters.
    flags: String,
}

/// Parse and validate a W3C `traceparent`.
///
/// Returns `None` for anything not matching
/// `<version>-<trace-id>-<span-id>-<flags>` with lowercase hex
/// fields, an all-zero trace-id or span-id, or the forbidden
/// version `ff`.
fn parse_traceparent(value: &str) -> Option<InboundTrace> {
    let mut parts = value.split('-');
    let version = parts.next()?;
    let trace_id = parts.next()?;
    let span_id = parts.next()?;
    let flags = parts.next()?;

    if parts.next().is_some() {
        return None;
    }

    if version.len() != 2 || !is_lower_hex(version) || version == "ff" {
        return None;
    }
    if trace_id.len() != TRACE_ID_LEN || !is_lower_hex(trace_id) || is_all_zero(trace_id) {
        return None;
    }
    if span_id.len() != SPAN_ID_LEN || !is_lower_hex(span_id) || is_all_zero(span_id) {
        return None;
    }
    if flags.len() != 2 || !is_lower_hex(flags) {
        return None;
    }

    Some(InboundTrace {
        trace_id: trace_id.to_owned(),
        flags: flags.to_owned(),
    })
}

/// Generate a 32-hex-character trace-id.
///
/// The core ID generator emits exactly 32 hex characters
/// (`{micros:012x}{seed:08x}{seq:012x}`), which is the W3C
/// trace-id width.
fn generate_trace_id(ctx: &HttpFilterContext<'_>) -> String {
    let id = ctx.id_generator.generate(ctx.time_source);
    if id.len() == TRACE_ID_LEN && is_lower_hex(&id) && !is_all_zero(&id) {
        return id;
    }
    // Defensive: keep emitting a well-formed trace-id if the core
    // generator's format ever changes.
    format!("{id:0>TRACE_ID_LEN$.TRACE_ID_LEN$}").replace(|c: char| !c.is_ascii_hexdigit(), "0")
}

/// Generate a 16-hex-character span-id for one hop.
///
/// The generator's trailing sequence counter advances on every
/// call, so each hop's span-id differs even when drawn within the
/// same microsecond.
fn generate_span_id(ctx: &HttpFilterContext<'_>) -> String {
    let id = generate_trace_id(ctx);
    id.get(TRACE_ID_LEN - SPAN_ID_LEN..).unwrap_or("").to_owned()
}

/// Check that every character is a lowercase hex digit.
fn is_lower_hex(value: &str) -> bool {
    value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Check that a hex string is entirely zeros.
fn is_all_zero(value: &str) -> bool {
    value.bytes().all(|b| b == b'0')
}
