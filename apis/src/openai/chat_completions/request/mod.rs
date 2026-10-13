// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Chat Completions request-body fact filter.
//!
//! Publishes the model fact for native `POST /v1/chat/completions` requests so a
//! downstream consumer such as `time_to_first_token` can label metrics by model.
//!
//! Request-head identity comes from `ai_operation`'s typed [`AiOperationMatch`]:
//! this filter consumes it and acts only on the body-bearing `createChatCompletion`
//! operation, then reads the model from the buffered body. The request head
//! remains authoritative for protocol identity; this filter only publishes
//! body facts for downstream consumers.
//!
//! The filter is transparent: a body that fails to classify simply yields no model
//! fact — it never rejects the request — so native Chat traffic forwards unchanged.
//!
//! [`AiOperationMatch`]: crate::operation_classifier::AiOperationMatch

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::{debug, trace};

use self::config::{ChatCompletionsRequestConfig, FILTER_NAME, build_config};
use crate::{
    classifier::{AiRequestFormat, ClassifiedRequest, classify_request_body},
    openai::chat_completions::routes as chat_completions_routes,
    operation_classifier::AiOperationMatch,
    promotion::is_promotable_value,
};

// -----------------------------------------------------------------------------
// OpenaiChatCompletionsRequestFilter
// -----------------------------------------------------------------------------

/// Publishes body facts for native Chat Completions create requests to durable
/// metadata and filter results.
///
/// # YAML
///
/// ```yaml
/// filter: openai_chat_completions_request
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_chat_completions_request
/// max_body_bytes: 1048576
/// ```
pub struct OpenaiChatCompletionsRequestFilter {
    /// Parsed and validated configuration.
    config: ChatCompletionsRequestConfig,
}

impl OpenaiChatCompletionsRequestFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ChatCompletionsRequestConfig = parse_filter_config(FILTER_NAME, config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for OpenaiChatCompletionsRequestFilter {
    fn name(&self) -> &'static str {
        // A string literal (not the `FILTER_NAME` const) so the filter-doc
        // generator's source scanner can extract the registered name.
        "openai_chat_completions_request"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
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

        if !is_create_chat_completion(ctx) {
            trace!(
                method = %ctx.request.method,
                path = ctx.request.uri.path(),
                "not a Chat Completions create request; leaving the model fact unset"
            );
            return Ok(FilterAction::Release);
        }

        let bytes = match body.as_ref() {
            Some(b) => b.as_ref(),
            None => &[],
        };

        let classified = classify_request_body(bytes);

        debug!(
            model = ?classified.model,
            stream = ?classified.stream,
            "classified chat completions request body"
        );

        write_metadata(ctx, &classified);
        promote_filter_results(ctx, &classified)?;

        Ok(FilterAction::Release)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Whether `ai_operation` matched this request head to the Chat Completions
/// create call.
///
/// The head is authoritative: only `createChatCompletion` carries an inference
/// body, so the model fact is produced for it alone. The list, get, update,
/// delete, and messages operations share the `openai_chat_completions` protocol
/// but are not inference calls and are skipped.
fn is_create_chat_completion(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions.get::<AiOperationMatch>().is_some_and(|matched| {
        chat_completions_routes::is_create_chat_completion(matched.application_protocol, matched.operation_id)
    })
}

/// Write durable metadata under the filter's namespace.
///
/// The format is fixed to the Chat Completions endpoint because the head already
/// proved the operation; only the model and stream flag are read from the body.
fn write_metadata(ctx: &mut HttpFilterContext<'_>, classified: &ClassifiedRequest) {
    ctx.set_metadata(
        "openai_chat_completions_request.format",
        AiRequestFormat::ChatCompletions.as_str(),
    );

    if let Some(model) = &classified.model
        && is_promotable_value(model)
    {
        ctx.set_metadata("openai_chat_completions_request.model", model.clone());
    }

    if let Some(stream) = classified.stream {
        ctx.set_metadata(
            "openai_chat_completions_request.stream",
            if stream { "true" } else { "false" },
        );
    }
}

/// Promote classification facts to filter results for branch conditions.
fn promote_filter_results(ctx: &mut HttpFilterContext<'_>, classified: &ClassifiedRequest) -> Result<(), FilterError> {
    let results = ctx.filter_results.entry("openai_chat_completions_request").or_default();

    results.set("format", AiRequestFormat::ChatCompletions.as_str())?;

    if let Some(model) = &classified.model
        && is_promotable_value(model)
    {
        results.set("model", model.clone())?;
    }

    if let Some(stream) = classified.stream {
        results.set("stream", if stream { "true" } else { "false" })?;
    }

    Ok(())
}
