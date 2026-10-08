// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-head classification of supported AI operations.
//!
//! The filter identifies an operation from the HTTP method, normalized path,
//! and protocol headers alone. The request body is never read, buffered, or
//! modified, so the result is available before any body-handling decision is
//! made.
//!
//! The same head-only classification runs from two hooks. The request-header
//! hook is the usual path. The request-body hook exists only so the match is
//! published before a downstream filter's buffered body pre-read: core runs that
//! pre-read ahead of the header phase, so a consumer that parses the body (for
//! example `openai_responses_request` under [`BodyMode::StreamBuffer`]) would
//! otherwise run before this classifier and find no match. Ordered ahead of that
//! consumer with [`BodyAccess::ReadOnly`] and [`BodyMode::Stream`], this filter's
//! body hook inspects only the head, publishes once, and returns
//! [`FilterAction::Continue`] without touching the body bytes. A completion
//! marker keeps classification to a single pass across both hooks and every
//! chunk, including when no operation matched. Enrolling in the body phase does
//! mean the body hook is invoked once per streamed chunk; the marker makes every
//! call after the first a cheap extension lookup, so the cost is a map read per
//! chunk, not reclassification.
//!
//! This ordering guarantees classification before a downstream filter's body
//! pre-read, not before all of core's pre-read work. Core enforces the
//! configured request body size limit while reading, so an oversized body can be
//! rejected with `413 Payload Too Large` before the first body hook runs and
//! thus before this filter classifies. That is the weaker guarantee; the
//! stronger one — classification ahead of every pre-read step — needs the core
//! hook point tracked in praxis#1142 and is not provided here.
//!
//! Every protocol-owned registry is consulted through the shared matcher, so
//! OpenAI and Anthropic operations are recognized by one filter. The classifier
//! holds no provider-specific matching logic: a registry is added by listing it
//! here, not by branching on its provider.
//!
//! A matched operation is published three ways: a typed
//! [`AiOperationMatch`] in request extensions for downstream filters,
//! including registry body metadata and allocation-free path-parameter
//! locations; metadata and filter results for branching; and optional
//! proxy-owned routing headers applied to the upstream request.
//!
//! The headers are pending mutations applied when the request is forwarded, so
//! the `router` filter — which matches the downstream request headers — does not
//! see them within the same header phase. Pipelines that branch on the
//! classification use `on_result` against the published filter results, which
//! are visible immediately. See the `openai/operation-classifier.yaml` example.
//!
//! Unmatched requests are left otherwise unchanged. Whether they are rejected,
//! forwarded to a fallback, or handled some other way is a routing policy
//! decision this filter does not make.

mod config;

#[cfg(test)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    ErrorResponseFormatterHandle, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::{BodyAccess, BodyMode},
    parse_filter_config,
};
use tracing::debug;

use self::config::{OperationClassifierConfig, ValidatedConfig, build_config};
use crate::{
    anthropic::routes as anthropic_messages_routes,
    openai::{
        chat_completions::routes as chat_completions_routes, conversations::routes as conversations_routes,
        responses::routes as responses_routes,
    },
    operation::{ApplicationProtocol, OperationEntry, PathParameterOffsets, RequestBody, RouteParams, Transport},
};

/// Filter name as configured in a pipeline.
const FILTER_NAME: &str = "ai_operation";

/// Classifies supported AI operations from the request head.
pub struct AiOperationFilter {
    /// Validated configuration.
    config: ValidatedConfig,
}

impl AiOperationFilter {
    /// Create the filter from parsed YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when configuration is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: OperationClassifierConfig = parse_filter_config(FILTER_NAME, config)?;
        let validated = build_config(&cfg)?;
        Ok(Box::new(Self { config: validated }))
    }

    /// Overwrite the configured routing headers with proxy-owned values.
    ///
    /// Uses set rather than append semantics so a client-supplied value of the
    /// same name cannot survive alongside the classifier's own.
    fn set_routing_headers(&self, ctx: &mut HttpFilterContext<'_>, matched: AiOperationMatch) {
        if let Some(name) = &self.config.application_protocol_header
            && let Ok(value) = http::HeaderValue::from_str(matched.application_protocol.as_str())
        {
            ctx.request_headers_to_set.push((name.clone(), value));
        }
        if let Some(name) = &self.config.operation_header
            && let Ok(value) = http::HeaderValue::from_str(matched.operation_id)
        {
            ctx.request_headers_to_set.push((name.clone(), value));
        }
    }

    /// Remove the configured routing headers from an unmatched request.
    ///
    /// An unmatched request carries no proxy-owned operation, so any value a
    /// client supplied under these names is stripped rather than forwarded.
    fn remove_routing_headers(&self, ctx: &mut HttpFilterContext<'_>) {
        if let Some(name) = &self.config.application_protocol_header {
            ctx.request_headers_to_remove.push(name.clone());
        }
        if let Some(name) = &self.config.operation_header {
            ctx.request_headers_to_remove.push(name.clone());
        }
    }

    /// Classify the request head and publish the result exactly once.
    ///
    /// Shared by the request-header and request-body hooks so the two cannot
    /// diverge. The completion marker makes the step idempotent: whichever hook
    /// runs first does the work, and later calls — the other hook, or a repeated
    /// body chunk — return without reclassifying. The marker is set even when no
    /// operation matched, so an unmatched request is classified once rather than
    /// on every chunk.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when publishing a matched operation fails.
    fn classify_and_publish(&self, ctx: &mut HttpFilterContext<'_>) -> Result<(), FilterError> {
        if ctx.extensions.get::<OperationClassified>().is_some() {
            return Ok(());
        }

        let method = ctx.request.method.as_str();
        let transport = request_transport(method, &ctx.request.headers);
        let path = ctx.request.uri.path();

        if let Some(matched) = classify(method, path, transport) {
            debug!(
                method,
                path,
                application_protocol = matched.application_protocol.as_str(),
                operation_id = matched.operation_id,
                transport = matched.transport.as_str(),
                "classified AI operation"
            );
            publish_match(ctx, matched)?;
            install_error_formatter(ctx, matched.application_protocol);
            self.set_routing_headers(ctx, matched);
        } else {
            debug!(method, path, transport = transport.as_str(), "no AI operation matched");
            self.remove_routing_headers(ctx);
        }

        ctx.extensions.insert(OperationClassified);
        Ok(())
    }
}

/// One operation classified from a request head.
///
/// Stored in request extensions so downstream filters share one authoritative
/// operation identity rather than re-deriving it from the same method and path.
/// Every field is `'static`; path parameters are stored as byte offsets into
/// the immutable request path and can be borrowed again without cloning it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AiOperationMatch {
    /// Application protocol that owns the operation, for example
    /// `openai_responses`.
    pub application_protocol: ApplicationProtocol,

    /// Stable operation ID.
    pub operation_id: &'static str,

    /// Transport the operation was reached over.
    pub transport: Transport,

    /// Registry-derived runtime request-body shape.
    pub request_body: RequestBody,

    /// Checked byte ranges for parameters captured from the request path.
    pub path_parameters: PathParameterOffsets,
}

/// Marks that this filter has run its one classify-and-publish pass.
///
/// Classification reads only the request head, which both hooks can see, so the
/// work must happen once however the pipeline schedules them. This zero-sized
/// extension records that it has, so the request-header hook, the request-body
/// hook, and every repeated body chunk resolve to a single pass — whether or not
/// an operation matched.
#[derive(Clone, Copy)]
struct OperationClassified;

#[async_trait]
impl HttpFilter for AiOperationFilter {
    fn name(&self) -> &'static str {
        "ai_operation"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        self.classify_and_publish(ctx)?;
        Ok(FilterAction::Continue)
    }

    /// Read-only: the head carries the whole classification, so the body is only
    /// observed, never buffered or modified. Declaring access enrolls the filter
    /// in the body phase so its hook runs there.
    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// Streamed: chunks pass through untouched. The filter never needs the full
    /// body, so it must not upgrade the pipeline-wide delivery mode to
    /// [`BodyMode::StreamBuffer`].
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    /// The body hook classifies from the request head, so the original method,
    /// path, and headers must be preserved into the body phase.
    fn needs_request_context(&self) -> bool {
        true
    }

    /// Publish the match from the body phase for a downstream buffered pre-read.
    ///
    /// A consumer that reads the body with [`BodyMode::StreamBuffer`] triggers a
    /// pre-read that core runs before the header phase. Ordered ahead of that
    /// consumer, this hook publishes the match so the consumer sees it. The body
    /// is never read, buffered, or altered, and the completion marker keeps this
    /// to one classification across the pre-read, the header phase, and every
    /// chunk.
    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let _ = (body, end_of_stream);
        self.classify_and_publish(ctx)?;
        Ok(FilterAction::Continue)
    }
}

/// Install the OpenAI error formatter for a matched OpenAI protocol.
///
/// Proxy-generated failures are returned in the shape the matched protocol's
/// clients expect rather than as RFC 9457 problem details.
///
/// Keyed off the operation resolved from the request head, so it does not
/// require a body, a successful parse, or any protocol-specific filter later
/// in the chain. A Chat Completions request therefore keeps OpenAI-shaped
/// errors without a Responses filter present, and a request whose body fails
/// to parse still gets them.
///
/// The head is the only place this can be decided for a bodyless operation.
/// `GET /v1/messages/batches/{id}` carries nothing to classify, so a
/// body-format filter can never install its formatter; without this, an
/// unreachable upstream would answer an Anthropic client with RFC 9457 problem
/// details.
///
/// Each protocol family is matched by its identifier prefix rather than an
/// enumerated list, so adding a registry to an existing family needs no change
/// here. A vendor protocol that shares neither prefix keeps the default shape.
fn install_error_formatter(ctx: &mut HttpFilterContext<'_>, protocol: ApplicationProtocol) {
    if protocol.as_str().starts_with("openai_") {
        ctx.extensions.insert(ErrorResponseFormatterHandle::new(
            crate::openai::error_response_formatter::OpenAiErrorFormatter,
        ));
    } else if protocol.as_str().starts_with("anthropic_") {
        // Built before the insert so the immutable borrow of the request head
        // ends before `extensions` is borrowed mutably.
        let formatter = crate::anthropic::error_response_formatter::AnthropicErrorFormatter::from_request_headers(
            &ctx.request.headers,
        );
        ctx.extensions.insert(ErrorResponseFormatterHandle::new(formatter));
    }
}

/// Publish a match as request extensions, metadata, and filter results.
///
/// The extension carries the generic identity for downstream filters, the
/// metadata is for logging and tracing, and the filter results are what
/// `on_result` branch conditions evaluate.
fn publish_match(ctx: &mut HttpFilterContext<'_>, matched: AiOperationMatch) -> Result<(), FilterError> {
    let application_protocol = matched.application_protocol.as_str();

    ctx.extensions.insert(matched);
    ctx.set_metadata("ai_operation.application_protocol", application_protocol);
    ctx.set_metadata("ai_operation.operation_id", matched.operation_id);

    let results = ctx.filter_results.entry(FILTER_NAME).or_default();
    results.set("application_protocol", application_protocol)?;
    results.set("operation_id", matched.operation_id)?;

    Ok(())
}

// -----------------------------------------------------------------------------
// Classification
// -----------------------------------------------------------------------------

/// Match a request head against every registered protocol.
///
/// HTTP-only registries are consulted in turn, so adding one is a new arm in
/// this list rather than a provider-specific branch anywhere else. Responses is
/// matched separately because transport is part of its operation identity: it is
/// the only protocol reachable over a `WebSocket` upgrade.
///
/// Registered path spaces do not overlap — OpenAI serves `/v1/responses`,
/// `/v1/conversations`, and `/v1/chat/completions` while Anthropic serves
/// `/v1/messages` — so at most one registry can match one head, and the order
/// below does not decide between protocols.
pub(crate) fn classify(method: &str, path: &str, transport: Transport) -> Option<AiOperationMatch> {
    let http_match = (transport == Transport::Http).then(|| {
        classify_conversation(method, path)
            .or_else(|| classify_chat_completions(method, path))
            .or_else(|| classify_anthropic_messages(method, path))
    });
    http_match
        .flatten()
        .or_else(|| classify_responses(method, path, transport))
}

/// Match one Conversations operation.
fn classify_conversation(method: &str, path: &str) -> Option<AiOperationMatch> {
    conversations_routes::match_route(method, path).and_then(|route| classified(route.spec, route.params, path))
}

/// Match one Chat Completions operation.
fn classify_chat_completions(method: &str, path: &str) -> Option<AiOperationMatch> {
    chat_completions_routes::match_route(method, path).and_then(|route| classified(route.spec, route.params, path))
}

/// Match one Anthropic Messages operation.
fn classify_anthropic_messages(method: &str, path: &str) -> Option<AiOperationMatch> {
    anthropic_messages_routes::match_route(method, path).and_then(|route| classified(route.spec, route.params, path))
}

/// Match one Responses operation.
fn classify_responses(method: &str, path: &str, transport: Transport) -> Option<AiOperationMatch> {
    responses_routes::match_route(method, path, transport).and_then(|route| classified(route.spec, route.params, path))
}

/// Build a published match from one registry entry and its parameters.
///
/// Reads the entry through [`OperationEntry`], so every registry — whatever
/// provider-specific wrapper it uses — is published identically and no provider
/// type appears in this module.
fn classified<T>(entry: &T, params: RouteParams<'_>, path: &str) -> Option<AiOperationMatch>
where
    T: OperationEntry,
{
    let spec = entry.spec();
    Some(AiOperationMatch {
        application_protocol: spec.application_protocol,
        operation_id: spec.operation_id,
        transport: spec.transport,
        request_body: spec.request_body,
        path_parameters: params.offsets_in(path)?,
    })
}

/// Determine the transport a request arrived over.
///
/// A `WebSocket` handshake is a `GET` carrying the opening handshake from
/// [RFC 6455 Section 4.1]. The method check is part of that definition, not an
/// optimization: without it, upgrade headers attached to a non-`GET` request
/// such as `POST /v1/responses` would select `WebSocket` transport and leave an
/// otherwise valid operation unclassified. `Connection` is a token list per
/// [RFC 9110 Section 7.6.1], so comma-separated and repeated field lines both
/// count. Exactly one `Upgrade` value is accepted, so a request nominating
/// several protocols is not treated as a `WebSocket` handshake.
///
/// [RFC 6455 Section 4.1]: https://datatracker.ietf.org/doc/html/rfc6455#section-4.1
/// [RFC 9110 Section 7.6.1]: https://datatracker.ietf.org/doc/html/rfc9110#section-7.6.1
fn request_transport(method: &str, headers: &http::HeaderMap) -> Transport {
    if method != http::Method::GET.as_str() {
        return Transport::Http;
    }

    let connection_upgrades = headers
        .get_all(http::header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|token| token.trim().eq_ignore_ascii_case("upgrade"));

    let mut upgrade_values = headers.get_all(http::header::UPGRADE).iter();
    let upgrades_to_websocket = upgrade_values
        .next()
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("websocket"))
        && upgrade_values.next().is_none();

    if connection_upgrades && upgrades_to_websocket {
        Transport::WebSocket
    } else {
        Transport::Http
    }
}
