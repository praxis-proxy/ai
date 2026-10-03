// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`AiGuardrailsFilter`] implementation and `HttpFilter` trait impl.

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::json_body::replace_json_body;
#[cfg(feature = "openai-responses")]
use praxis_ai_apis::openai::{local_tool_guardrail_messages, record_local_tool_guardrail_failure};
use praxis_core::{
    config::InsecureOptions,
    subrequest::{DEPTH_HEADER, SubRequestClient},
};
#[cfg(test)]
use praxis_filter::parse_filter_config;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, FilterPipeline, HttpFilter, HttpFilterContext, IterationState,
    Rejection, SubrequestRuntime,
};

use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    providers::{GuardCalloutRuntime, GuardPhase, GuardResult, MessageRedaction, nemo},
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
/// Every provider callout runs through Praxis's filtered-subrequest executor.
/// The optional `outbound_chain` adds destination-bound authentication,
/// authorization, audit, and static service credentials; when omitted it
/// defaults to an empty pass-through chain. Parent and child contexts stay
/// isolated; user-scoped credential projection is handled separately in #880.
///
/// Because this filter reads the request body before the header-phase
/// security filters on the main chain run, operators should treat the
/// pre-read body as untrusted input and configure an outbound chain whenever
/// the provider requires destination-bound policy enforcement.
///
/// **Wire format:** client request/response evaluation supports Chat
/// Completions only (`messages` on requests, `choices[].message` on
/// responses). `phase.tool_results` separately evaluates canonical local
/// Responses tool results inside an IRR step. Other Responses content,
/// Anthropic Messages, and MCP wire bodies are not supported yet (see ai#1043).
/// For `phase.tool_results`, a provider `modified` verdict fails closed with
/// `502 guardrail_error`; sanitized text is not forwarded because it cannot
/// yet be safely mapped back to the canonical tool-result items. Only a
/// `passed` verdict permits model re-entry.
///
/// For `NeMo`, `provider.guardrails.config_ids` selects deployed guardrail
/// configurations. When `provider.guardrails` is omitted, the request omits
/// `config_ids` so the service can use its default configuration. Omit
/// `provider.model` to leave the selected configuration's models unchanged;
/// a non-empty value replaces or adds its main model.
///
/// # YAML configuration
///
/// ```yaml
/// filter: ai_guardrails
/// outbound_chain: nemo-outbound # optional
/// provider:
///   type: nemo
///   endpoint: "http://nemo:8000/v1/checks"
///   guardrails:
///     config_ids: ["your-config"]
///   timeout_ms: 5000
/// phase:
///   request: true
///   response: true
///   tool_results: false
/// ```
pub struct AiGuardrailsFilter {
    /// Guard provider instance.
    provider: nemo::NemoProvider,
    /// Which phases to evaluate.
    phase: PhaseConfig,
    /// Prebuilt outbound filter chain for provider callouts.
    outbound: Arc<FilterPipeline>,
    /// Per-callout deadline derived from the provider configuration.
    callout_timeout: std::time::Duration,
    /// Chosen verdict for tests that must not call `NeMo`.
    #[cfg(test)]
    scripted_verdict: Option<GuardResult>,
}

impl AiGuardrailsFilter {
    /// Build a filter from parsed config, a bound outbound chain, and a
    /// shared sub-request client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or provider validation fails.
    pub(crate) fn build(
        config: AiGuardrailsConfig,
        outbound: Arc<FilterPipeline>,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        #[cfg(not(feature = "openai-responses"))]
        if config.phase.tool_results {
            return Err("ai_guardrails: phase.tool_results requires the openai-responses feature".into());
        }

        let (provider, callout_timeout) = match config.provider.provider_type {
            ProviderType::Nemo => {
                let provider = nemo::NemoProvider::from_config(&config.provider.config, client)?;
                let timeout = provider.callout_timeout();
                (provider, timeout)
            },
        };

        Ok(Box::new(Self {
            provider,
            phase: config.phase,
            outbound,
            callout_timeout,
            #[cfg(test)]
            scripted_verdict: None,
        }))
    }

    /// Create a filter from parsed YAML config.
    ///
    /// Production pipelines must register `ai_guardrails` through
    /// [`register_chain_binding`](praxis_filter::FilterRegistry::register_chain_binding)
    /// so the outbound chain is resolved at construction time. This
    /// constructor exists for tests and builds a permissive outbound test
    /// pipeline directly.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    #[cfg(test)]
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client = crate::isolated_subrequest_client(4);
        Self::from_config_with_client(config, client)
    }

    /// Create a filter using the shared [`SubRequestClient`].
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    #[cfg(test)]
    pub(crate) fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AiGuardrailsConfig = parse_filter_config("ai_guardrails", config)?;
        let outbound = test_outbound_chain()?;
        Self::build(cfg, outbound, client)
    }

    /// Test-only constructor so fail-closed redaction can be exercised
    /// without a live `NeMo` call. `NeMo` skips `/v1/checks` when there
    /// is no phase-target message, which would otherwise never produce
    /// [`GuardResult::Redact`].
    #[cfg(test)]
    pub(super) fn with_verdict(verdict: GuardResult, phase: PhaseConfig) -> Result<Self, FilterError> {
        let client = crate::isolated_subrequest_client(4);
        let config = serde_yaml::from_str("endpoint: \"http://127.0.0.1:9/v1/checks\"")
            .map_err(|error| -> FilterError { format!("ai_guardrails (nemo): {error}").into() })?;
        Ok(Self {
            provider: nemo::NemoProvider::from_config(&config, client)?,
            phase,
            outbound: test_outbound_chain()?,
            callout_timeout: std::time::Duration::from_secs(5),
            scripted_verdict: Some(verdict),
        })
    }

    /// Capture downstream identity, nesting, deadline, and the bound chain for a callout.
    fn callout_runtime(&self, ctx: &HttpFilterContext<'_>) -> GuardCalloutRuntime<'_> {
        let now = Instant::now();
        let deadline = effective_callout_deadline(
            now,
            self.callout_timeout,
            ctx.extensions.get::<IterationState>().map(IterationState::deadline),
        );

        GuardCalloutRuntime {
            downstream: SubrequestRuntime::new(
                ctx.client_addr,
                ctx.downstream_tls,
                ctx.peer_identity.clone(),
                ctx.request_start,
            ),
            depth: subrequest_depth(ctx),
            deadline,
            outbound: &self.outbound,
        }
    }

    /// Apply a scripted test verdict, or call the configured `NeMo` provider.
    async fn evaluate(
        &self,
        messages: Vec<serde_json::Value>,
        phase: GuardPhase,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<GuardResult, FilterError> {
        #[cfg(test)]
        if let Some(verdict) = &self.scripted_verdict {
            return Ok(verdict.clone());
        }
        self.provider.evaluate(messages, phase, runtime).await
    }

    /// Evaluate the local result suffix and leave terminal response ownership
    /// with `openai_agentic_loop`.
    #[cfg(feature = "openai-responses")]
    async fn evaluate_local_tool_results(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        messages: Vec<serde_json::Value>,
    ) -> Result<FilterAction, FilterError> {
        let runtime = self.callout_runtime(ctx);
        match self.evaluate(messages, GuardPhase::Request, &runtime).await {
            Ok(result) => record_local_tool_verdict(ctx, result),
            Err(error) => {
                tracing::error!(%error, phase = "tool_results", "ai_guardrails: evaluation failed");
                record_local_tool_evaluation_failure(ctx, &error.to_string())
            },
        }
    }
}

#[cfg(test)]
/// Build a permissive, minimal outbound pipeline for direct unit tests.
fn test_outbound_chain() -> Result<Arc<FilterPipeline>, FilterError> {
    use praxis_core::config::FilterEntry;
    use praxis_filter::FilterRegistry;

    let registry = FilterRegistry::with_builtins();
    let mut entries: Vec<FilterEntry> = serde_yaml::from_str("- filter: request_id\n")
        .map_err(|error| -> FilterError { format!("ai_guardrails: {error}").into() })?;
    let mut pipeline = FilterPipeline::build(&mut entries, &registry)?;
    pipeline.set_allow_private_upstreams(true);
    Ok(Arc::new(pipeline))
}

#[async_trait]
impl HttpFilter for AiGuardrailsFilter {
    fn name(&self) -> &'static str {
        "ai_guardrails"
    }

    fn visit_nested_pipelines(&mut self, visitor: &mut dyn FnMut(&mut FilterPipeline)) {
        if let Some(pipeline) = Arc::get_mut(&mut self.outbound) {
            visitor(pipeline);
        } else {
            debug_assert!(false, "outbound pipeline must be uniquely owned during configuration");
        }
    }

    fn referenced_files(&self) -> Vec<std::path::PathBuf> {
        self.outbound.referenced_files()
    }

    fn apply_insecure_options(&self, options: &InsecureOptions) {
        self.outbound.apply_insecure_options(options);
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
        if self.phase.request {
            BodyAccess::ReadWrite
        } else if self.phase.tool_results {
            BodyAccess::ReadOnly
        } else {
            BodyAccess::None
        }
    }

    fn request_body_mode(&self) -> BodyMode {
        if self.phase.request {
            BodyMode::StreamBuffer {
                max_bytes: Some(DEFAULT_MAX_BODY_BYTES),
            }
        } else {
            // Tool-result evaluation reads canonical Responses state at EOS;
            // it must not impose the Chat request body's 1 MiB buffer cap on
            // the surrounding IRR step.
            BodyMode::Stream
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

        #[cfg(feature = "openai-responses")]
        if self.phase.tool_results {
            match local_tool_guardrail_messages(&ctx.extensions, DEFAULT_MAX_BODY_BYTES) {
                Ok(messages) if !messages.is_empty() => {
                    return self.evaluate_local_tool_results(ctx, messages).await;
                },
                Err(error) => {
                    return record_local_tool_evaluation_failure(ctx, &error.to_string());
                },
                Ok(_) => {},
            }
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
        let runtime = self.callout_runtime(ctx);
        let result = self.evaluate(messages, GuardPhase::Request, &runtime).await?;
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

    #[expect(
        clippy::too_many_lines,
        reason = "keeps response evaluation and fail-closed mapping together"
    )]
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
            let runtime = self.callout_runtime(ctx);
            // `on_response_body` is sync (Pingora constraint); use `block_in_place`
            // to bridge into async. See #51 for the plan to make this truly async.
            tokio::task::block_in_place(|| handle.block_on(self.evaluate(messages, GuardPhase::Response, &runtime)))
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

/// Record one completed local tool-result verdict.
#[cfg(feature = "openai-responses")]
pub(super) fn record_local_tool_verdict(
    ctx: &mut HttpFilterContext<'_>,
    result: GuardResult,
) -> Result<FilterAction, FilterError> {
    match result {
        GuardResult::Pass => {
            set_verdict(ctx, "passed")?;
            tracing::debug!(phase = "tool_results", "ai_guardrails: verdict passed");
        },
        GuardResult::Block { reason } => {
            set_verdict(ctx, "blocked")?;
            let message = format!("Tool result blocked by guardrails: {reason}");
            tracing::warn!(phase = "tool_results", %reason, "ai_guardrails: verdict blocked");
            if !record_local_tool_guardrail_failure(&mut ctx.extensions, 403, "content_blocked", message.clone()) {
                return Ok(FilterAction::Reject(Rejection::status(403).with_body(message)));
            }
        },
        GuardResult::Redact { reason, .. } => {
            set_verdict(ctx, "redacted")?;
            let message = format!(
                "Tool result was modified by guardrails but sanitized content cannot be safely applied: {reason}"
            );
            tracing::error!(phase = "tool_results", %reason, "ai_guardrails: modified result rejected");
            if !record_local_tool_guardrail_failure(&mut ctx.extensions, 502, "guardrail_error", message.clone()) {
                return Ok(FilterAction::Reject(Rejection::status(502).with_body(message)));
            }
        },
    }
    Ok(FilterAction::Continue)
}

/// Fail closed through the Responses loop owner when tool-result evaluation
/// cannot complete.
#[cfg(feature = "openai-responses")]
fn record_local_tool_evaluation_failure(
    ctx: &mut HttpFilterContext<'_>,
    reason: &str,
) -> Result<FilterAction, FilterError> {
    let message = format!("Tool-result guardrail evaluation failed: {reason}");
    if record_local_tool_guardrail_failure(&mut ctx.extensions, 502, "guardrail_error", message.clone()) {
        return Ok(FilterAction::Continue);
    }
    Err(message.into())
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Extract the current filtered-subrequest depth from trusted runtime state.
fn subrequest_depth(ctx: &HttpFilterContext<'_>) -> u8 {
    resolve_subrequest_depth(
        ctx.extensions.get::<IterationState>().map(IterationState::depth),
        &ctx.request.headers,
    )
}

/// Resolve nesting depth, preferring IRR-owned state over the framework header.
fn resolve_subrequest_depth(iteration_state_depth: Option<u8>, headers: &http::HeaderMap) -> u8 {
    iteration_state_depth.unwrap_or_else(|| {
        headers
            .get(DEPTH_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(0)
    })
}

/// Bound the provider timeout by the enclosing IRR deadline, when present.
fn effective_callout_deadline(
    now: Instant,
    callout_timeout: std::time::Duration,
    iteration_deadline: Option<Instant>,
) -> Instant {
    let provider_deadline = now.checked_add(callout_timeout).unwrap_or(now);
    iteration_deadline.map_or(provider_deadline, |deadline| provider_deadline.min(deadline))
}

/// Record the provider verdict in `ctx.filter_results` and map it to
/// the corresponding [`FilterAction`].
fn record_verdict(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    result: GuardResult,
    phase: GuardPhase,
) -> Result<FilterAction, FilterError> {
    let verdict = result.status_label();
    let phase_label = phase.label();
    set_verdict(ctx, verdict)?;

    match result {
        GuardResult::Pass => {
            tracing::debug!(verdict, phase = phase_label, "ai_guardrails: verdict");
            Ok(FilterAction::Continue)
        },
        GuardResult::Block { reason } => Ok(enforce_block(body, reason, phase, phase_label, verdict)),
        GuardResult::Redact { replacements, reason } => {
            tracing::warn!(verdict, phase = phase_label, %reason, "ai_guardrails: verdict");
            apply_redaction(body, replacements, phase)
        },
    }
}

/// Rewrite the buffered body with each `NeMo` masked turn and continue.
///
/// Request-phase framing is repaired by core via `mutated_request_body_len`.
/// Response headers are already committed. A rewrite that fits is space-padded
/// to that length. A rewrite that would grow the body is refused: truncating
/// it would return malformed JSON with the original 200, so the client receives
/// a valid fail-closed document fitted to the committed length instead.
fn apply_redaction(
    body: &mut Option<Bytes>,
    replacements: Vec<MessageRedaction>,
    phase: GuardPhase,
) -> Result<FilterAction, FilterError> {
    match phase {
        GuardPhase::Request => {
            apply_request_redaction(body, replacements)?;
            Ok(FilterAction::Continue)
        },
        GuardPhase::Response => {
            if let Err(error) = apply_response_redaction(body, replacements) {
                tracing::error!(%error, "ai_guardrails: response-phase redaction failed");
                install_length_fitting_error(
                    body,
                    &format!("Guardrail redaction failed: {error}"),
                    "guardrail_error",
                    "evaluation_failed",
                );
            }
            Ok(FilterAction::Continue)
        },
    }
}

/// Replace each redacted user message's `content` with its masked text.
fn apply_request_redaction(body: &mut Option<Bytes>, replacements: Vec<MessageRedaction>) -> Result<(), FilterError> {
    if replacements.is_empty() {
        return Err("ai_guardrails: cannot redact: no message replacements".into());
    }
    let mut value = parse_json_body(body, "request")?;
    let Some(messages) = value.get_mut("messages").and_then(serde_json::Value::as_array_mut) else {
        return Err("ai_guardrails: request body does not contain recognizable messages".into());
    };
    apply_message_replacements(messages, replacements, "user")?;
    replace_json_body(body, &value, "ai_guardrails", "messages")
        .map_err(|e| -> FilterError { format!("ai_guardrails: failed to serialize redacted body: {e}").into() })?;
    Ok(())
}

/// Replace each redacted choice message's `content` when the rewrite fits.
///
/// A longer document is refused. The caller emits a fail-closed body instead
/// of truncating the chat completion to the committed `Content-Length`.
fn apply_response_redaction(body: &mut Option<Bytes>, replacements: Vec<MessageRedaction>) -> Result<(), FilterError> {
    if replacements.is_empty() {
        return Err("ai_guardrails: cannot redact: no message replacements".into());
    }
    let mut value = parse_json_body(body, "response")?;
    let Some(choices) = value.get_mut("choices").and_then(serde_json::Value::as_array_mut) else {
        return Err("ai_guardrails: response body does not contain recognizable choices".into());
    };
    for replacement in replacements {
        let index = replacement.index;
        let Some(choice) = choices.get_mut(index) else {
            return Err(format!("ai_guardrails: cannot redact: choice index {index} has no message").into());
        };
        redact_choice(choice, replacement)?;
    }
    let serialized = serde_json::to_string(&value)
        .map_err(|e| -> FilterError { format!("ai_guardrails: failed to serialize redacted body: {e}").into() })?;
    let original_len = body.as_ref().map_or(0, Bytes::len);
    if serialized.len() > original_len {
        return Err("ai_guardrails: redacted response exceeds committed Content-Length".into());
    }
    *body = Some(fit_to_committed_length(serialized, body));
    Ok(())
}

/// Rewrite one choice message and drop `logprobs` that still quote the original tokens.
///
/// An absent `logprobs` field is left absent. Inserting it would grow the body past the committed `Content-Length`.
fn redact_choice(choice: &mut serde_json::Value, replacement: MessageRedaction) -> Result<(), FilterError> {
    let Some(message) = choice.get_mut("message") else {
        return Err(format!(
            "ai_guardrails: cannot redact: choice index {} has no message",
            replacement.index
        )
        .into());
    };
    set_message_content(message, replacement.modified_text)?;
    let Some(choice) = choice.as_object_mut() else {
        return Err("ai_guardrails: choice is not a JSON object".into());
    };
    if choice.contains_key("logprobs") {
        choice.insert("logprobs".to_owned(), serde_json::Value::Null);
    }
    Ok(())
}

/// Apply each replacement to `messages[index]`, requiring `expected_role`.
fn apply_message_replacements(
    messages: &mut [serde_json::Value],
    replacements: Vec<MessageRedaction>,
    expected_role: &str,
) -> Result<(), FilterError> {
    for replacement in replacements {
        let Some(message) = messages.get_mut(replacement.index) else {
            return Err(format!(
                "ai_guardrails: cannot redact: message index {} is out of range",
                replacement.index
            )
            .into());
        };
        if message.get("role").and_then(serde_json::Value::as_str) != Some(expected_role) {
            return Err(format!(
                "ai_guardrails: cannot redact: message {} is not a {expected_role} turn",
                replacement.index
            )
            .into());
        }
        set_message_content(message, replacement.modified_text)?;
    }
    Ok(())
}

/// Parse a buffered JSON body, labeling errors as `kind` (`request` or `response`).
fn parse_json_body(body: &Option<Bytes>, kind: &str) -> Result<serde_json::Value, FilterError> {
    let Some(raw) = body.as_ref() else {
        return Err(format!("ai_guardrails: cannot redact a missing {kind} body").into());
    };
    serde_json::from_slice(raw)
        .map_err(|e| -> FilterError { format!("ai_guardrails: {kind} body is not valid JSON: {e}").into() })
}

/// Set a chat message's `content` field to `modified_text`.
fn set_message_content(message: &mut serde_json::Value, modified_text: String) -> Result<(), FilterError> {
    let Some(object) = message.as_object_mut() else {
        return Err("ai_guardrails: message is not a JSON object".into());
    };
    if object
        .get("content")
        .is_some_and(|content| !content.is_string() && !content.is_null())
    {
        return Err("ai_guardrails: cannot redact: message content is not a string".into());
    }
    object.insert("content".to_owned(), serde_json::Value::String(modified_text));
    Ok(())
}

/// Publish the guardrail verdict for routing and observability.
fn set_verdict(ctx: &mut HttpFilterContext<'_>, verdict: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry("ai_guardrails")
        .or_default()
        .set("status", verdict)?;
    Ok(())
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
fn replace_body_with_error(body: &mut Option<Bytes>, message: &str, error_type: &str, code: &str) {
    let error_json = error_document(message, error_type, code);
    *body = Some(fit_to_committed_length(error_json, body));
}

/// Replace the response body with a valid error document of the committed length.
///
/// Response headers, including `Content-Length`, are already on the wire. The
/// message is shortened until the document fits, then padded with spaces.
/// Trailing spaces are JSON whitespace, so the client still parses one value.
fn install_length_fitting_error(body: &mut Option<Bytes>, message: &str, error_type: &str, code: &str) {
    let original_len = body.as_ref().map_or(0, Bytes::len);
    let full = error_document(message, error_type, code);
    if full.len() > original_len {
        tracing::warn!(
            new_len = full.len(),
            original_len,
            "ai_guardrails: fail-closed response shortened to fit committed Content-Length",
        );
    }
    *body = Some(length_fitting_error(message, error_type, code, original_len));
}

/// OpenAI-style error JSON with `message`, `type`, and `code`.
fn error_document(message: &str, error_type: &str, code: &str) -> String {
    serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    })
    .to_string()
}

/// Error JSON that parses and occupies exactly `original_len` bytes.
///
/// Prefers the full message, then the longest message prefix that fits, then a
/// code-only object, then `{}`. A body shorter than two bytes cannot hold a
/// JSON object, so it is filled with spaces and does not echo the upstream text.
pub(super) fn length_fitting_error(message: &str, error_type: &str, code: &str, original_len: usize) -> Bytes {
    if original_len == 0 {
        return Bytes::new();
    }
    if let Some(document) = longest_error_document(message, error_type, code, original_len) {
        return pad_to_len(document, original_len);
    }
    for document in fallback_error_documents(code) {
        if document.len() <= original_len {
            return pad_to_len(document, original_len);
        }
    }
    Bytes::from(vec![b' '; original_len])
}

/// Longest `error_document` whose message is a prefix of `message` and whose
/// serialized form is at most `original_len` bytes.
fn longest_error_document(message: &str, error_type: &str, code: &str, original_len: usize) -> Option<String> {
    let full = error_document(message, error_type, code);
    if full.len() <= original_len {
        return Some(full);
    }

    let chars: Vec<char> = message.chars().collect();
    let mut low = 0;
    let mut high = chars.len();
    let mut best = None;
    while low <= high {
        let mid = low.midpoint(high);
        let prefix: String = chars.iter().take(mid).collect();
        let candidate = error_document(&prefix, error_type, code);
        if candidate.len() <= original_len {
            best = Some(candidate);
            if mid == high {
                break;
            }
            low = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }
    best
}

/// Documents used when even an empty message does not fit.
fn fallback_error_documents(code: &str) -> [String; 2] {
    [
        serde_json::json!({"error": {"code": code}}).to_string(),
        "{}".to_owned(),
    ]
}

/// Pad `text` with trailing spaces out to `len`.
fn pad_to_len(text: String, len: usize) -> Bytes {
    let mut bytes = text.into_bytes();
    debug_assert!(
        bytes.len() <= len,
        "pad_to_len must not truncate; a longer document would no longer be valid JSON"
    );
    bytes.resize(len, b' ');
    Bytes::from(bytes)
}

/// Fit `replacement` bytes to the original response body length.
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
fn extract_messages(body: &Bytes) -> Result<Vec<serde_json::Value>, FilterError> {
    let mut json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| -> FilterError { format!("ai_guardrails: request body is not valid JSON: {e}").into() })?;

    if let Some(messages) = json.get_mut("messages").filter(|m| m.is_array())
        && let serde_json::Value::Array(messages) = std::mem::take(messages)
    {
        return Ok(messages);
    }

    Err("ai_guardrails: request body does not contain recognizable messages".into())
}

/// Extract assistant messages from an OpenAI Chat Completion response body.
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

#[cfg(test)]
mod depth_tests {
    use std::time::{Duration, Instant};

    use http::{HeaderMap, HeaderValue};

    use super::{DEPTH_HEADER, effective_callout_deadline, resolve_subrequest_depth};

    #[test]
    fn depth_prefers_iteration_state_over_header() {
        let mut headers = HeaderMap::new();
        headers.insert(DEPTH_HEADER, HeaderValue::from_static("7"));
        assert_eq!(resolve_subrequest_depth(Some(3), &headers), 3);
    }

    #[test]
    fn depth_falls_back_to_framework_header() {
        let mut headers = HeaderMap::new();
        headers.insert(DEPTH_HEADER, HeaderValue::from_static("4"));
        assert_eq!(resolve_subrequest_depth(None, &headers), 4);
    }

    #[test]
    fn depth_defaults_to_zero_without_trusted_state() {
        assert_eq!(resolve_subrequest_depth(None, &HeaderMap::new()), 0);
    }

    #[test]
    fn callout_deadline_uses_provider_timeout_without_iteration() {
        let now = Instant::now();
        assert_eq!(
            effective_callout_deadline(now, Duration::from_secs(5), None),
            now + Duration::from_secs(5)
        );
    }

    #[test]
    fn callout_deadline_is_capped_by_iteration() {
        let now = Instant::now();
        let iteration_deadline = now + Duration::from_secs(2);
        assert_eq!(
            effective_callout_deadline(now, Duration::from_secs(5), Some(iteration_deadline)),
            iteration_deadline
        );
    }
}
