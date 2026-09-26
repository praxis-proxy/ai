// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Single request-body processor for body-bearing Responses operations.
//!
//! Operation identity comes from the request head through the Responses
//! registry, so the body is never inspected to decide whether this filter
//! applies. A matched request is then deserialized exactly once, and
//! that one parsed value produces every downstream fact: the classification
//! metadata, the promoted headers and filter results, the proxy-owned
//! identifiers, and [`ResponsesState`].
//!
//! Create requests with `background=true` are rejected, because Praxis does not
//! implement the asynchronous Responses lifecycle.
//!
//! This replaces the pair of `openai_responses_format` and
//! `openai_responses_validate`. Those two each parsed the
//! same body independently, so routing facts, proxy-owned defaults, and state
//! could be derived from different parses of one request.
//!
//! Metadata and filter results keep the `openai_responses_format` namespace.
//! Twelve downstream filters read those keys, and renaming them is a separate
//! change rather than a side effect of consolidating the parse.
//!
//! # YAML
//!
//! ```yaml
//! filter: openai_responses_request
//! on_invalid: reject
//! headers:
//!   format: x-praxis-ai-format
//!   model: x-praxis-ai-model
//!   stream: x-praxis-ai-stream
//! ```

#[cfg(test)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
};
use tracing::{debug, trace};

use super::{
    config::{ResponsesFormatConfig, build_config},
    error::responses_error_rejection_with_code,
    extract_conversation_id,
    routes::{self as responses_routes},
    state::ResponsesState,
};
use crate::{
    classifier::{AiRequestFormat, ClassifiedRequest, classify_object, empty_result},
    operation::{RequestBody, Transport},
};

/// Filter name as configured in a pipeline.
const FILTER_NAME: &str = "openai_responses_request";

/// Processes a Responses request body once and initializes state.
///
/// Replaces the `openai_responses_format` and `openai_responses_validate` pair.
/// Configuration is unchanged from `openai_responses_format`, so a chain that
/// ran both swaps them for this one filter and keeps the same `on_invalid` and
/// `headers` settings.
///
/// The operation is recognized from the request head, and the registry decides
/// which operations carry a body worth parsing: create, compact, and input
/// token counts. Bodyless operations — fetch, delete, cancel, list input items,
/// and the `WebSocket` handshake — are released untouched, as is Conversations
/// API traffic. `on_invalid` governs only bodies that fail to parse.
///
/// Rejects `background=true` with a 400, matching `openai_responses_format`,
/// because Praxis does not implement the asynchronous Responses lifecycle.
///
/// Promotes `openai_responses_format.*` metadata and filter results, and
/// generates `responses.response_id` (`resp_` + 32 hex chars, CSPRNG),
/// `responses.conversation_id`, `responses.store`, `responses.background`, and
/// `responses.stream`.
pub struct OpenaiResponsesRequestFilter {
    /// Classification and promotion configuration.
    config: ResponsesFormatConfig,
}

impl OpenaiResponsesRequestFilter {
    /// Create the filter from parsed YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when configuration is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ResponsesFormatConfig = parse_filter_config(FILTER_NAME, config)?;
        let validated = build_config(FILTER_NAME, cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for OpenaiResponsesRequestFilter {
    fn name(&self) -> &'static str {
        "openai_responses_request"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(body_shape) = matched_request_body(ctx) else {
            trace!(
                method = %ctx.request.method,
                path = ctx.request.uri.path(),
                "not a body-bearing Responses operation"
            );
            return Ok(FilterAction::Release);
        };

        // An operation whose body the specification marks optional is complete
        // without one, so an absent body is not an invalid body and must not
        // reach `on_invalid`.
        if !body_shape.is_required() && body.as_deref().is_none_or(<[u8]>::is_empty) {
            trace!(
                path = ctx.request.uri.path(),
                "optional request body absent, nothing to parse"
            );
            return Ok(FilterAction::Release);
        }

        // The one parse feeds classification, promotion, and state alike. A body
        // that cannot be classified follows `on_invalid` instead.
        let (parsed, classified) = match parse_and_classify_create_body(body) {
            Ok(pair) => pair,
            Err(format) => return handle_unclassifiable(ctx, format, &self.config),
        };

        if let Some(action) = super::handle_unsupported_background(&classified) {
            return Ok(action);
        }

        if let Some(action) = reject_conflicting_history_selectors(&parsed) {
            return Ok(action);
        }

        publish_request_facts(ctx, &classified, parsed, &self.config)?;

        Ok(FilterAction::Release)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Publish everything the one parse produced.
///
/// Kept out of `on_request_body` so the filter entry point stays a readable
/// sequence of guards.
///
/// # Errors
///
/// Returns [`FilterError`] when a filter result cannot be published.
fn publish_request_facts(
    ctx: &mut HttpFilterContext<'_>,
    classified: &ClassifiedRequest,
    parsed: serde_json::Value,
    config: &ResponsesFormatConfig,
) -> Result<(), FilterError> {
    let mode = super::compute_mode(classified);

    // Classification is published for every body, whatever it turned out to
    // be, exactly as the standalone classifier did.
    publish_classification(ctx, classified, config, mode)?;

    // Proxy-owned identifiers and `ResponsesState` are Responses-only. A body
    // positively identified as another protocol keeps that identity and must
    // not gain Responses state, or state-driven filters such as the agentic
    // loop would pick up traffic the previous validation stage released
    // untouched.
    if classified.format != AiRequestFormat::Responses {
        trace!(
            format = classified.format.as_str(),
            "classified as another protocol, leaving Responses state uninitialized"
        );
        return Ok(());
    }

    let response_id = format!("resp_{}", ctx.id_generator.generate(ctx.time_source));
    let conversation_id = resolve_conversation_id(ctx, &parsed);

    enrich_context(ctx, classified, &response_id, &conversation_id);
    insert_responses_state(ctx, parsed, &response_id);

    debug!(
        response_id = %response_id,
        conversation_id = %conversation_id,
        mode = ?mode,
        "create request processed and state initialized"
    );

    Ok(())
}

/// Publish the classification facts for one body.
///
/// Shared by the classified path and the unclassifiable one so both publish the
/// same keys, as the standalone classifier did for every body it saw.
///
/// # Errors
///
/// Returns [`FilterError`] when a filter result cannot be published.
fn publish_classification(
    ctx: &mut HttpFilterContext<'_>,
    classified: &ClassifiedRequest,
    config: &ResponsesFormatConfig,
    mode: Option<&'static str>,
) -> Result<(), FilterError> {
    super::install_error_formatter(ctx, classified.format);
    super::write_metadata(ctx, classified, mode);
    super::promote_headers(ctx, classified, config, mode);
    super::promote_filter_results(ctx, classified, mode)
}

/// Apply `on_invalid` to a body that could not be classified.
///
/// `reject` and `error` return the configured rejection. `continue` forwards
/// the request, but still publishes the `invalid_json` or `non_json` facts, so
/// a chain that routes or branches on those keys behaves as it did before this
/// filter replaced the classifier and validator pair.
///
/// # Errors
///
/// Returns [`FilterError`] when a filter result cannot be published.
fn handle_unclassifiable(
    ctx: &mut HttpFilterContext<'_>,
    format: AiRequestFormat,
    config: &ResponsesFormatConfig,
) -> Result<FilterAction, FilterError> {
    if let Some(action) = super::handle_invalid_format(format, config) {
        debug!(format = format.as_str(), "rejecting unclassifiable create body");
        return Ok(action);
    }

    let classified = empty_result(format);
    let mode = super::compute_mode(&classified);
    publish_classification(ctx, &classified, config, mode)?;

    debug!(
        format = format.as_str(),
        "forwarding unclassifiable create body under on_invalid: continue"
    );
    Ok(FilterAction::Release)
}

/// Classify a body that the request head already identified as Responses.
///
/// The matched operation is authoritative over body heuristics. A valid create
/// body may omit every discriminator those heuristics look for —
/// `{"model":"gpt-5"}` is a legitimate create request — and would otherwise be
/// published as `unknown`, which makes downstream Responses filters skip it and
/// lets `background: true` past a rejection that keys off the published format.
///
/// Only unknown classifications are upgraded, so a body positively identified
/// as another format keeps that identity and its own handling.
fn classify_matched_operation(obj: &serde_json::Map<String, serde_json::Value>) -> ClassifiedRequest {
    let mut classified = classify_object(obj);
    if classified.format == AiRequestFormat::UnknownJson {
        classified.format = AiRequestFormat::Responses;
    }
    classified
}

/// The declared request-body shape when this is a body-bearing operation.
///
/// Resolved from the request head through the shared registry — the same source
/// of truth the `openai_operation` classifier uses — so no body heuristic
/// decides whether this filter applies, and the filter works whether or not the
/// classifier is present in the chain.
///
/// The registry already records which operations declare a request body and
/// whether it is required, so that declaration selects what is worth parsing
/// rather than a hand-written path list that can drift from it. Bodyless
/// operations — fetch, delete, cancel, list input items, and the `WebSocket`
/// handshake — yield `None` and are released untouched.
fn matched_request_body(ctx: &HttpFilterContext<'_>) -> Option<RequestBody> {
    responses_routes::match_route(ctx.request.method.as_str(), ctx.request.uri.path(), Transport::Http)
        .map(|route| route.spec.request_body())
        .filter(|shape| shape.is_present())
}

/// Parse a create body once and extract its routing facts.
///
/// Returns the parsed value alongside its classification so the caller holds
/// both from a single deserialization. A missing body is an empty slice rather
/// than an error, and a top-level value that is not an object is
/// `invalid_json`, both matching the classifier this filter replaces.
///
/// # Errors
///
/// Returns the format to publish when the body cannot be classified.
fn parse_and_classify_create_body(
    body: &Option<Bytes>,
) -> Result<(serde_json::Value, ClassifiedRequest), AiRequestFormat> {
    let bytes: &[u8] = body.as_deref().unwrap_or(&[]);
    let parsed = serde_json::from_slice::<serde_json::Value>(bytes).map_err(|_ignored| {
        if bytes.is_empty() {
            AiRequestFormat::NonJson
        } else {
            AiRequestFormat::InvalidJson
        }
    })?;
    let classified = classify_matched_operation(parsed.as_object().ok_or(AiRequestFormat::InvalidJson)?);
    Ok((parsed, classified))
}

/// Reject requests that select both supported sources of conversation history.
fn reject_conflicting_history_selectors(body: &serde_json::Value) -> Option<FilterAction> {
    let conflicts = body.get("previous_response_id").is_some_and(|value| !value.is_null())
        && body.get("conversation").is_some_and(|value| !value.is_null());
    conflicts.then(|| {
        FilterAction::Reject(responses_error_rejection_with_code(
            400,
            "invalid_request_error",
            "mutually_exclusive_parameters",
            "Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.",
        ))
    })
}

/// Extract or generate a conversation ID for the request.
fn resolve_conversation_id(ctx: &HttpFilterContext<'_>, body: &serde_json::Value) -> String {
    if let Some(id) = extract_conversation_id(body) {
        trace!(conversation_id = %id, "conversation ID extracted from request");
        id
    } else {
        let id = format!("conv_{}", ctx.id_generator.generate(ctx.time_source));
        trace!(conversation_id = %id, "conversation ID generated");
        id
    }
}

/// Publish validated request facts for downstream filters.
///
/// Reads the parsed classification directly rather than round-tripping through
/// classifier metadata, so these values cannot disagree with the body they came
/// from. Spec defaults apply: `store` defaults to true, the others to false.
fn enrich_context(
    ctx: &mut HttpFilterContext<'_>,
    classified: &ClassifiedRequest,
    response_id: &str,
    conversation_id: &str,
) {
    ctx.set_metadata("responses.response_id", response_id);
    ctx.set_metadata("responses.conversation_id", conversation_id);

    let store = classified.store.unwrap_or(true);
    let background = classified.background.unwrap_or(false);
    let stream = classified.stream.unwrap_or(false);

    ctx.set_metadata("responses.store", if store { "true" } else { "false" });
    ctx.set_metadata("responses.background", if background { "true" } else { "false" });
    ctx.set_metadata("responses.stream", if stream { "true" } else { "false" });

    trace!(store, background, stream, "request facts published");
}

/// Initialize canonical request state, including metadata that must survive IRR steps.
fn insert_responses_state(ctx: &mut HttpFilterContext<'_>, parsed: serde_json::Value, response_id: &str) {
    let mut state = ResponsesState::from_request_body(parsed);
    state.response_id = Some(response_id.to_owned());
    ctx.extensions.insert(state);
}
