// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-head classification of supported OpenAI operations.
//!
//! The filter identifies an operation from the HTTP method, normalized path,
//! and protocol headers alone. No request body is read or buffered, so the
//! result is available before any body-handling decision is made.
//!
//! A matched operation is published three ways: a typed
//! [`OpenAiOperationMatch`] in request extensions for downstream filters,
//! metadata and filter results for branching, and optional proxy-owned routing
//! headers applied to the upstream request.
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
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use tracing::debug;

use self::config::{OperationClassifierConfig, ValidatedConfig, build_config};
use crate::openai::{
    conversations::routes as conversations_routes,
    operation::{OpenAiApiFamily, OpenAiTransport},
    responses::routes as responses_routes,
};

/// Filter name as configured in a pipeline.
const FILTER_NAME: &str = "openai_operation";

/// Classifies supported OpenAI operations from the request head.
pub struct OpenaiOperationFilter {
    /// Validated configuration.
    config: ValidatedConfig,
}

impl OpenaiOperationFilter {
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
    fn set_routing_headers(&self, ctx: &mut HttpFilterContext<'_>, matched: OpenAiOperationMatch) {
        if let Some(name) = &self.config.application_protocol_header
            && let Ok(value) = http::HeaderValue::from_str(matched.family.application_protocol())
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
}

/// One operation classified from a request head.
///
/// Stored in request extensions so downstream filters share one authoritative
/// operation identity rather than re-deriving it from the same method and path.
/// Every field is `'static`; borrowed path parameters remain available through
/// each family's own matcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenAiOperationMatch {
    /// API family that owns the operation.
    ///
    /// Published downstream as its provider-qualified application protocol,
    /// for example `openai_responses`, rather than as the bare family name.
    pub family: OpenAiApiFamily,

    /// Stable operation ID.
    pub operation_id: &'static str,

    /// Transport the operation was reached over.
    pub transport: OpenAiTransport,
}

#[async_trait]
impl HttpFilter for OpenaiOperationFilter {
    fn name(&self) -> &'static str {
        "openai_operation"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let method = ctx.request.method.as_str();
        let transport = request_transport(method, &ctx.request.headers);
        let path = ctx.request.uri.path();

        let Some(matched) = classify(method, path, transport) else {
            debug!(
                method,
                path,
                transport = transport.as_str(),
                "no OpenAI operation matched"
            );
            self.remove_routing_headers(ctx);
            return Ok(FilterAction::Continue);
        };

        debug!(
            method,
            path,
            application_protocol = matched.family.application_protocol(),
            operation_id = matched.operation_id,
            transport = matched.transport.as_str(),
            "classified OpenAI operation"
        );

        publish_match(ctx, matched)?;
        self.set_routing_headers(ctx, matched);

        Ok(FilterAction::Continue)
    }
}

/// Publish a match as request extensions, metadata, and filter results.
///
/// The extension carries the typed identity for downstream filters, the
/// metadata is for logging and tracing, and the filter results are what
/// `on_result` branch conditions evaluate.
fn publish_match(ctx: &mut HttpFilterContext<'_>, matched: OpenAiOperationMatch) -> Result<(), FilterError> {
    let application_protocol = matched.family.application_protocol();

    ctx.extensions.insert(matched);
    ctx.set_metadata("openai_operation.application_protocol", application_protocol);
    ctx.set_metadata("openai_operation.operation_id", matched.operation_id);

    let results = ctx.filter_results.entry(FILTER_NAME).or_default();
    results.set("application_protocol", application_protocol)?;
    results.set("operation_id", matched.operation_id)?;

    Ok(())
}

// -----------------------------------------------------------------------------
// Classification
// -----------------------------------------------------------------------------

/// Match a request head against every registered API family.
///
/// Families are consulted in registration order. Their path spaces do not
/// overlap, so at most one can match a given method and path.
fn classify(method: &str, path: &str, transport: OpenAiTransport) -> Option<OpenAiOperationMatch> {
    if let Some(route) = conversations_routes::match_route(method, path) {
        // Conversations is reached over plain HTTP only.
        if transport == OpenAiTransport::Http {
            return Some(OpenAiOperationMatch {
                family: route.spec.family,
                operation_id: route.spec.operation_id,
                transport: route.spec.transport,
            });
        }
    }

    if let Some(route) = responses_routes::match_route(method, path, transport) {
        return Some(OpenAiOperationMatch {
            family: route.spec.family,
            operation_id: route.spec.operation_id,
            transport: route.spec.transport,
        });
    }

    None
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
fn request_transport(method: &str, headers: &http::HeaderMap) -> OpenAiTransport {
    if method != http::Method::GET.as_str() {
        return OpenAiTransport::Http;
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
        OpenAiTransport::WebSocket
    } else {
        OpenAiTransport::Http
    }
}
