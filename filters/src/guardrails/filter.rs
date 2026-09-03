// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`AiGuardrailsFilter`] implementation and `HttpFilter` trait impl.

use async_trait::async_trait;
use bytes::Bytes;
use praxis_core::subrequest::{SubRequestClient, SubRequestConnector};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};

use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    providers::{GuardPhase, GuardProvider, GuardResult, nemo::NemoProvider},
};

/// Maximum request body size to buffer (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

// -----------------------------------------------------------------------------
// AiGuardrailsFilter
// -----------------------------------------------------------------------------

/// Calls an external AI guardrail provider to evaluate request and
/// response bodies. The provider determines whether content should
/// be passed, blocked, or redacted.
///
/// # YAML configuration
///
/// ```yaml
/// filter: ai_guardrails
/// provider:
///   type: nemo
///   endpoint: "http://nemo:8000/v1/guardrail/checks"
///   timeout_ms: 5000
/// phase:
///   request: true
///   response: true
/// ```
///
/// # Example
///
/// ```ignore
/// use praxis_ai_filters::AiGuardrailsFilter;
///
/// let yaml: serde_yaml::Value = serde_yaml::from_str(
///     r#"
/// provider:
///   type: nemo
///   endpoint: "http://nemo:8000/v1/guardrail/checks"
/// "#,
/// )
/// .unwrap();
/// let filter = AiGuardrailsFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "ai_guardrails");
/// ```
pub struct AiGuardrailsFilter {
    /// Guard provider instance.
    provider: Box<dyn GuardProvider>,
    /// Which phases to evaluate.
    phase: PhaseConfig,
}

impl AiGuardrailsFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// Uses an isolated [`SubRequestClient`] with a default pool
    /// size of 4. Prefer [`from_config_with_client`] when a shared
    /// client is available.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    /// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
    /// [`from_config_with_client`]: Self::from_config_with_client
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client = SubRequestClient::new(SubRequestConnector::new(4, None));
        Self::build(config, client)
    }

    /// Create a filter using the shared [`SubRequestClient`].
    ///
    /// The shared client inherits the server-level pool size and
    /// connection limits from the runtime configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    /// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Self::build(config, client)
    }

    /// Shared constructor body for [`from_config`](Self::from_config) and
    /// [`from_config_with_client`](Self::from_config_with_client).
    fn build(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AiGuardrailsConfig = parse_filter_config("ai_guardrails", config)?;

        let provider: Box<dyn GuardProvider> = match cfg.provider.provider_type {
            ProviderType::Nemo => Box::new(NemoProvider::from_config(&cfg.provider.config, client)?),
        };

        Ok(Box::new(Self {
            provider,
            phase: cfg.phase,
        }))
    }
}

#[async_trait]
impl HttpFilter for AiGuardrailsFilter {
    fn name(&self) -> &'static str {
        "ai_guardrails"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if self.phase.response && !is_event_stream(ctx) {
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(DEFAULT_MAX_BODY_BYTES),
            });
        }
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(DEFAULT_MAX_BODY_BYTES),
        }
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

        if !self.phase.request {
            return Ok(FilterAction::Continue);
        }

        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        if bytes.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let messages = extract_messages(bytes)?;
        let result = self.provider.evaluate(messages, GuardPhase::Request).await?;
        record_verdict(ctx, body, result, GuardPhase::Request)
    }

    fn response_body_access(&self) -> BodyAccess {
        if self.phase.response {
            BodyAccess::ReadWrite
        } else {
            BodyAccess::None
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream || !self.phase.response {
            return Ok(FilterAction::Continue);
        }

        // Only evaluate when the body was fully buffered (StreamBuffer).
        // SSE / streaming responses stay in Stream mode and are not evaluated.
        if !matches!(ctx.response_body_mode, BodyMode::StreamBuffer { .. }) {
            tracing::debug!("ai_guardrails: skipping response-phase evaluation (body not buffered)");
            return Ok(FilterAction::Continue);
        }

        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        if bytes.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let evaluation = extract_response_messages(bytes).and_then(|messages| {
            let handle = tokio::runtime::Handle::current();
            // `on_response_body` is sync (Pingora constraint); use `block_in_place`
            // to bridge into async. See #51 for the plan to make this truly async.
            tokio::task::block_in_place(|| handle.block_on(self.provider.evaluate(messages, GuardPhase::Response)))
        });

        match evaluation {
            Ok(result) => record_verdict(ctx, body, result, GuardPhase::Response),
            Err(e) => {
                tracing::error!(error = %e, "ai_guardrails: response-phase evaluation failed");
                replace_body_with_error(
                    body,
                    &format!("Guardrail evaluation failed: {e}"),
                    "guardrail_error",
                    "evaluation_failed",
                );
                Ok(FilterAction::Continue)
            },
        }
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Record the provider verdict in `ctx.filter_results` and map it to
/// the corresponding [`FilterAction`].
///
/// The `phase` parameter controls how a `Block` verdict is enforced:
///
/// - **Request phase**: returns `FilterAction::Reject(403)` - headers have not been sent yet, so a clean 403 is
///   possible.
///
/// - **Response phase**: response headers (including the upstream's 200 status and `Content-Length`) are already
///   committed by the time `on_response_body` runs.  A `Reject(403)` would be converted to a 500 by Pingora (see
///   `praxis-proxy/pingora` issue #51).  Instead, the response body is replaced with a JSON error payload and padded to
///   the original `Content-Length` so Pingora does not report `PrematureBodyEnd`.  JSON parsers ignore trailing ASCII
///   spaces, so clients parse the error cleanly.
fn record_verdict(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    result: GuardResult,
    phase: GuardPhase,
) -> Result<FilterAction, FilterError> {
    let verdict = result.status_label();
    let phase_label = phase.label();
    ctx.filter_results
        .entry("ai_guardrails")
        .or_default()
        .set("status", verdict)?;

    match result {
        GuardResult::Pass => {
            tracing::debug!(verdict, phase = phase_label, "ai_guardrails: verdict");
            Ok(FilterAction::Continue)
        },
        GuardResult::Block { reason } => Ok(enforce_block(body, reason, phase, phase_label, verdict)),
        GuardResult::Redact { reason, .. } => {
            tracing::warn!(verdict, phase = phase_label, %reason, "ai_guardrails: verdict; forwarding unchanged until #579");
            Ok(FilterAction::Continue)
        },
    }
}

/// Enforce a `Block` verdict for the given phase.
fn enforce_block(
    body: &mut Option<Bytes>,
    reason: String,
    phase: GuardPhase,
    phase_label: &str,
    verdict: &str,
) -> FilterAction {
    match phase {
        GuardPhase::Request => {
            tracing::warn!(verdict, phase = phase_label, %reason, "ai_guardrails: verdict");
            FilterAction::Reject(Rejection::status(403).with_body(reason))
        },
        GuardPhase::Response => {
            tracing::warn!(verdict, phase = phase_label, %reason, "ai_guardrails: verdict - replacing body");
            replace_body_with_error(
                body,
                &format!("Response blocked by guardrails: {reason}"),
                "guardrail_violation",
                "content_blocked",
            );
            FilterAction::Continue
        },
    }
}

/// Replace the response body with an error JSON payload.
///
/// Used on the response side when headers are already committed and
/// the body is the only channel for communicating errors to the client.
fn replace_body_with_error(body: &mut Option<Bytes>, message: &str, error_type: &str, code: &str) {
    let error_json = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    })
    .to_string();
    *body = Some(fit_to_committed_length(error_json, body));
}

/// Fit `replacement` bytes to the original response body length.
///
/// The downstream `Content-Length` is committed by the time
/// `on_response_body` runs - praxis has no response-side equivalent of
/// `apply_mutated_content_length`. Emitting fewer bytes than
/// `Content-Length` causes Pingora to report `PrematureBodyEnd` and
/// abort the connection. Emitting more bytes is an HTTP/1.1 framing
/// desync.
///
/// Pads with trailing ASCII spaces on shrink (JSON parsers ignore them);
/// truncates on grow (safe failure mode - corrupts JSON but cannot cause
/// response smuggling).
pub(super) fn fit_to_committed_length(replacement: String, original_body: &Option<Bytes>) -> Bytes {
    let original_len = original_body.as_ref().map_or(0, Bytes::len);
    let replacement = replacement.into_bytes();
    match replacement.len().cmp(&original_len) {
        std::cmp::Ordering::Equal => Bytes::from(replacement),
        std::cmp::Ordering::Less => {
            let mut padded = replacement;
            padded.resize(original_len, b' ');
            Bytes::from(padded)
        },
        std::cmp::Ordering::Greater => {
            tracing::warn!(
                new_len = replacement.len(),
                original_len,
                "ai_guardrails: replacement body larger than committed Content-Length; truncating",
            );
            let prefix = replacement.get(..original_len).unwrap_or(&replacement);
            let safe = match std::str::from_utf8(prefix) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            let mut result = replacement;
            result.truncate(safe);
            result.resize(original_len, b' ');
            Bytes::from(result)
        },
    }
}

/// Extract messages from an OpenAI Chat Completion request body.
///
/// Supports:
/// - OpenAI Chat request: `{"messages": [...]}`
///
/// Returns an error for unrecognized body formats to prevent
/// silently skipping guardrail evaluation.
///
/// # Errors
///
/// Returns [`FilterError`] if the body is not valid JSON or does not
/// contain a recognizable messages field.
fn extract_messages(body: &Bytes) -> Result<Vec<serde_json::Value>, FilterError> {
    let mut json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| -> FilterError { format!("ai_guardrails: request body is not valid JSON: {e}").into() })?;

    // OpenAI Chat format: {"messages": [...]}
    if let Some(messages) = json.get_mut("messages").filter(|m| m.is_array())
        && let serde_json::Value::Array(messages) = std::mem::take(messages)
    {
        return Ok(messages);
    }

    Err("ai_guardrails: request body does not contain recognizable messages".into())
}

/// Extract assistant messages from an OpenAI Chat Completion response body.
///
/// Supports:
/// - OpenAI Chat Completion response: `{"choices": [{"message": {...}}]}`
///
/// Each `message` object from the `choices` array is returned as-is
/// so the guardrail provider sees the full assistant message
/// (role, content, `tool_calls`, etc.).
///
/// # Errors
///
/// Returns [`FilterError`] if the body is not valid JSON or does not
/// contain a recognizable choices/message structure.
fn extract_response_messages(body: &Bytes) -> Result<Vec<serde_json::Value>, FilterError> {
    let mut json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| -> FilterError { format!("ai_guardrails: response body is not valid JSON: {e}").into() })?;

    if let Some(choices) = json.get_mut("choices").and_then(|c| c.as_array_mut()) {
        let num_choices = choices.len();
        let messages: Vec<serde_json::Value> = choices
            .iter_mut()
            .filter_map(|c| c.get_mut("message").map(std::mem::take))
            .collect();
        if messages.is_empty() {
            return Err("ai_guardrails: response body does not contain recognizable choices".into());
        }
        if messages.len() != num_choices {
            return Err(format!(
                "ai_guardrails: {num_choices} choices but only {} contain a message field",
                messages.len(),
            )
            .into());
        }
        return Ok(messages);
    }

    Err("ai_guardrails: response body does not contain recognizable choices".into())
}

/// Whether the upstream response has a `text/event-stream` content type.
fn is_event_stream(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .and_then(|r| r.headers.get("content-type"))
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| {
            ct.split(';')
                .next()
                .is_some_and(|media| media.trim().eq_ignore_ascii_case("text/event-stream"))
        })
}
