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
    Rejection, SubrequestRuntime, body::MAX_JSON_BODY_BYTES,
};
use serde::Deserialize;

use super::{
    config::{AiGuardrailsConfig, PhaseConfig, ProviderType},
    providers::{GuardCalloutRuntime, GuardPhase, GuardResult, MessageRedaction, nemo},
};

/// Maximum request body size to buffer (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

/// Rejection body used when response guardrails cannot inspect streaming output.
const STREAMING_RESPONSE_GUARDRAIL_ERROR: &str =
    "ai_guardrails: streaming responses are not supported while a response guardrail is active";

// -----------------------------------------------------------------------------
// AiGuardrailsFilter
// -----------------------------------------------------------------------------

/// Calls an external AI guardrail provider to evaluate request and
/// response bodies. The provider determines whether content should
/// be passed, blocked, or redacted.
///
/// Registered as a security filter. Pipeline validation rejects
/// `failure_mode: open` unless `insecure_options.allow_open_security_filters`
/// is set. `insecure_options.skip_pipeline_validation` also bypasses this
/// check. With either option, a provider timeout or error can silently skip
/// screening.
///
/// Request `conditions:` are rejected for the same reason: a security filter
/// must not be skipped for non-matching requests. Body hooks do not run
/// inside a branch, so a router branch is not a substitute. A non-empty
/// request body without a Chat Completions `messages` array fails the
/// request with 500. Serve `/v1/responses` and `/v1/embeddings` from a
/// listener that does not include this filter; put `ai_guardrails` only on
/// a listener whose routes are Chat Completions.
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
/// A provider `modified` verdict on the request rewrites each targeted user
/// message and forwards the masked body. A request rewrite that cannot be
/// applied is rejected with HTTP 502 so the original text is not sent
/// upstream. On the response, each targeted choice is rebuilt from `index`,
/// `finish_reason`, and a message containing only `role` and the masked
/// `content`. A choice whose message has array `content`, `tool_calls`, or
/// `function_call` cannot be redacted; the client receives the original HTTP
/// 200 with an `evaluation_failed` error document. `on_response` drops
/// `Content-Length` so core reframes the buffered body at its natural size.
///
/// When `phase.response` is enabled, Chat Completions requests with
/// `"stream": true` are rejected with HTTP 422 until incremental SSE
/// response evaluation is supported (see #1108).
///
/// For `NeMo`, `provider.guardrails.config_ids` selects deployed guardrail
/// configurations. When `provider.guardrails` is omitted, the request omits
/// `config_ids` so the service can use its default configuration. Omit
/// `provider.model` to leave the selected configuration's models unchanged;
/// a non-empty value replaces or adds its main model.
///
/// In production, set `provider.endpoint` to `https://`. Set
/// `provider.tls.ca_file` to the PEM bundle that signed the `NeMo` server
/// certificate. That bundle is this peer's trust store, and hostname
/// verification stays enabled. Without it, the call uses the trust store
/// loaded for the Praxis process.
///
/// `provider.token_file` is optional. When set, Praxis reads that Kubernetes
/// login token at startup and sends it on the call to this `NeMo` address.
/// The token is not on the request the outbound chain sees. Praxis adds
/// `Authorization` after that chain runs, so a filter there cannot remove or
/// replace it. An `http` endpoint is refused unless its host is loopback.
/// A missing file refuses to start. If the file later disappears, the `NeMo`
/// call is refused and the chat is not sent to the model.
///
/// # YAML configuration
///
/// ```yaml
/// filter: ai_guardrails
/// outbound_chain: nemo-outbound # optional
/// provider:
///   type: nemo
///   endpoint: "https://nemo.default.svc:8443/v1/checks"
///   guardrails:
///     config_ids: ["your-config"]
///   tls:
///     ca_file: /var/run/secrets/nemo/ca.crt
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
    /// Production pipelines register `ai_guardrails` through
    /// [`register_chain_binding_with_class`](praxis_filter::FilterRegistry::register_chain_binding_with_class)
    /// as a security filter, so the outbound chain is resolved at construction
    /// time. This constructor exists for tests and builds a permissive outbound
    /// test pipeline directly.
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

#[expect(clippy::too_many_lines, reason = "implements the complete filter lifecycle")]
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
        if !self.phase.response {
            return Ok(FilterAction::Continue);
        }

        if is_event_stream(ctx) {
            return Ok(FilterAction::Reject(
                Rejection::status(502).with_body(STREAMING_RESPONSE_GUARDRAIL_ERROR),
            ));
        }

        remove_stale_content_length(ctx);
        ctx.set_response_body_mode(BodyMode::StreamBuffer {
            max_bytes: Some(DEFAULT_MAX_BODY_BYTES),
        });
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        if self.phase.request {
            BodyAccess::ReadWrite
        } else if self.phase.response || self.phase.tool_results {
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
        } else if self.phase.response {
            // Response-only guardrails buffer every request up to `MAX_JSON_BODY_BYTES`
            // before forwarding to inspect `stream`. They do not deserialize request messages.
            BodyMode::StreamBuffer {
                max_bytes: Some(MAX_JSON_BODY_BYTES),
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
        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        if bytes.is_empty() {
            return Ok(FilterAction::Continue);
        }

        if !self.phase.request {
            // The current response-phase guardrail implementation cannot inspect a
            // streamed response because SSE responses remain in streaming mode.
            // Rather than let it stream through unevaluated, reject the request
            // before the model is invoked.
            if self.phase.response {
                match requests_streaming(bytes) {
                    Ok(true) => {
                        return Ok(FilterAction::Reject(
                            Rejection::status(422).with_body(STREAMING_RESPONSE_GUARDRAIL_ERROR),
                        ));
                    },
                    Ok(false) => {},
                    Err(_) => return Ok(FilterAction::Reject(Rejection::status(400))),
                }
            }
            return Ok(FilterAction::Continue);
        }

        let request = parse_request(bytes)?;
        if self.phase.response && stream_requested(request.stream.as_ref()) {
            return Ok(FilterAction::Reject(
                Rejection::status(422).with_body(STREAMING_RESPONSE_GUARDRAIL_ERROR),
            ));
        }

        let messages = request.into_messages()?;
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
        GuardResult::Block { reason } => Ok(enforce_block(body, reason, phase, verdict)),
        GuardResult::Redact { replacements, reason } => {
            tracing::warn!(verdict, phase = phase_label, %reason, "ai_guardrails: verdict");
            Ok(apply_redaction(body, replacements, phase))
        },
    }
}

/// Rewrite the buffered body with each `NeMo` masked turn and continue.
///
/// Request-phase framing is repaired by core via `mutated_request_body_len`.
/// A request rewrite that cannot be applied is rejected with HTTP 502.
/// Returning an error would let `failure_mode: open` forward the original text.
///
/// Response headers stay committed, except `Content-Length`, which
/// `on_response` removes so core reframes the buffered body. A response
/// rewrite that cannot be applied replaces the body with an `evaluation_failed`
/// document at its natural size and keeps the original HTTP 200.
fn apply_redaction(body: &mut Option<Bytes>, replacements: Vec<MessageRedaction>, phase: GuardPhase) -> FilterAction {
    match phase {
        GuardPhase::Request => {
            if let Err(error) = apply_request_redaction(body, replacements) {
                tracing::error!(%error, "ai_guardrails: request-phase redaction failed");
                return FilterAction::Reject(
                    Rejection::status(502).with_body(format!("Guardrail redaction failed: {error}")),
                );
            }
            FilterAction::Continue
        },
        GuardPhase::Response => {
            if let Err(error) = apply_response_redaction(body, replacements) {
                tracing::error!(%error, "ai_guardrails: response-phase redaction failed");
                replace_body_with_error(
                    body,
                    &format!("Guardrail redaction failed: {error}"),
                    "guardrail_error",
                    "evaluation_failed",
                );
            }
            FilterAction::Continue
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

/// Replace each redacted choice with an allowlisted object.
///
/// The rewritten document is forwarded at its natural size. `on_response`
/// already dropped `Content-Length` so core can reframe it.
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
    *body = Some(Bytes::from(serialized));
    Ok(())
}

/// Rebuild one choice from `index`, `finish_reason`, and a message of `role`
/// plus masked `content`.
///
/// `logprobs`, `token_ids`, and `message.reasoning` repeat the original
/// completion, so they are dropped rather than forwarded.
fn redact_choice(choice: &mut serde_json::Value, replacement: MessageRedaction) -> Result<(), FilterError> {
    ensure_redactable_choice(choice, replacement.index)?;
    write_allowlisted_choice(choice, replacement.modified_text)
}

/// Confirm the choice is an object whose message has text `content`, a string
/// `role`, and no tool invocation.
///
/// `tool_calls` and `function_call` are not copied onto the rebuilt message.
/// Keeping `finish_reason` without them would tell the client a tool turn
/// completed and then omit the calls. Copying them could leak text `NeMo` just
/// masked, so a message that has either field fails closed.
fn ensure_redactable_choice(choice: &serde_json::Value, index: usize) -> Result<(), FilterError> {
    let Some(choice_object) = choice.as_object() else {
        return Err("ai_guardrails: choice is not a JSON object".into());
    };
    let Some(message) = choice_object.get("message") else {
        return Err(format!("ai_guardrails: cannot redact: choice index {index} has no message").into());
    };
    ensure_text_content(message)?;
    if message_has_tool_invocation(message) {
        return Err("ai_guardrails: cannot redact: message includes a tool invocation".into());
    }
    if message.get("role").is_some_and(serde_json::Value::is_string) {
        return Ok(());
    }
    Err("ai_guardrails: cannot redact: message role is not a string".into())
}

/// Whether `message` carries `tool_calls` or a legacy `function_call`.
///
/// A null field is treated as absent, matching text `content`.
fn message_has_tool_invocation(message: &serde_json::Value) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    object.get("tool_calls").is_some_and(|value| !value.is_null())
        || object.get("function_call").is_some_and(|value| !value.is_null())
}

/// `index`, `finish_reason`, and message `role` copied from one choice.
struct AllowlistedChoiceFields {
    /// Choice `index`, when the upstream object included it.
    index: Option<serde_json::Value>,
    /// Choice `finish_reason`, when the upstream object included it.
    finish_reason: Option<serde_json::Value>,
    /// Message `role`.
    role: serde_json::Value,
}

/// Replace `choice` with `index`, `finish_reason`, and a message of `role` plus `content`.
fn write_allowlisted_choice(choice: &mut serde_json::Value, modified_text: String) -> Result<(), FilterError> {
    let fields = allowlisted_choice_fields(choice)?;
    let mut message = serde_json::Map::new();
    message.insert("role".to_owned(), fields.role);
    message.insert("content".to_owned(), serde_json::Value::String(modified_text));

    let mut rebuilt = serde_json::Map::new();
    if let Some(index) = fields.index {
        rebuilt.insert("index".to_owned(), index);
    }
    rebuilt.insert("message".to_owned(), serde_json::Value::Object(message));
    if let Some(finish_reason) = fields.finish_reason {
        rebuilt.insert("finish_reason".to_owned(), finish_reason);
    }
    *choice = serde_json::Value::Object(rebuilt);
    Ok(())
}

/// Move the allowlisted fields out of `choice`.
fn allowlisted_choice_fields(choice: &mut serde_json::Value) -> Result<AllowlistedChoiceFields, FilterError> {
    let Some(choice_object) = choice.as_object_mut() else {
        return Err("ai_guardrails: choice is not a JSON object".into());
    };
    let index = choice_object.remove("index");
    let finish_reason = choice_object.remove("finish_reason");
    let Some(role) = choice_object
        .get_mut("message")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|message| message.remove("role"))
    else {
        return Err("ai_guardrails: cannot redact: message role is not a string".into());
    };
    Ok(AllowlistedChoiceFields {
        index,
        finish_reason,
        role,
    })
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
    ensure_text_content(message)?;
    let Some(object) = message.as_object_mut() else {
        return Err("ai_guardrails: message is not a JSON object".into());
    };
    object.insert("content".to_owned(), serde_json::Value::String(modified_text));
    Ok(())
}

/// Reject message `content` that is present and neither a string nor null.
fn ensure_text_content(message: &serde_json::Value) -> Result<(), FilterError> {
    let Some(object) = message.as_object() else {
        return Err("ai_guardrails: message is not a JSON object".into());
    };
    if object
        .get("content")
        .is_some_and(|content| !content.is_string() && !content.is_null())
    {
        return Err("ai_guardrails: cannot redact: message content is not a string".into());
    }
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
fn enforce_block(body: &mut Option<Bytes>, reason: String, phase: GuardPhase, verdict: &str) -> FilterAction {
    let phase_label = phase.label();
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

/// Replace the response body with an error JSON payload at its natural size.
fn replace_body_with_error(body: &mut Option<Bytes>, message: &str, error_type: &str, code: &str) {
    *body = Some(Bytes::from(error_document(message, error_type, code)));
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

/// Remove stale `Content-Length` so core recomputes framing for the buffered body.
///
/// The length on the upstream response is not the length the client will see:
/// compression, or a translator listed ahead of this filter, can drop it later.
/// The body is fully buffered here, so core can reframe from the bytes this
/// filter forwards.
fn remove_stale_content_length(ctx: &mut HttpFilterContext<'_>) {
    if let Some(response) = &mut ctx.response_header {
        response.headers.remove(http::header::CONTENT_LENGTH);
        ctx.response_headers_modified = true;
    }
}

/// Request fields needed by response-only guardrails.
#[derive(Deserialize)]
struct StreamingRequest {
    /// Whether the upstream should stream its response.
    #[serde(default)]
    stream: Option<serde_json::Value>,
}

/// Request fields needed by request-phase guardrails.
#[derive(Deserialize)]
struct GuardrailsRequest {
    /// Whether the upstream should stream its response.
    #[serde(default)]
    stream: Option<serde_json::Value>,
    /// Chat Completion messages evaluated by request guardrails.
    messages: Option<serde_json::Value>,
}

impl GuardrailsRequest {
    /// Extract the request messages for guardrail evaluation.
    fn into_messages(self) -> Result<Vec<serde_json::Value>, FilterError> {
        match self.messages {
            Some(serde_json::Value::Array(messages)) => Ok(messages),
            _ => Err("ai_guardrails: request body does not contain recognizable messages".into()),
        }
    }
}

/// Whether the stream field requests a streamed response.
fn stream_requested(stream: Option<&serde_json::Value>) -> bool {
    stream.and_then(serde_json::Value::as_bool).unwrap_or(false)
}

/// Whether the request asks the upstream for a streamed response via
/// `"stream": true`.
///
/// A malformed body returns an error, allowing the pipeline's failure mode
/// to handle it consistently with request-phase parsing.
fn requests_streaming(body: &Bytes) -> Result<bool, FilterError> {
    let request: StreamingRequest = serde_json::from_slice(body).map_err(|error| request_body_parse_error(&error))?;

    Ok(stream_requested(request.stream.as_ref()))
}

/// Parse request fields for request-phase guardrails in one pass.
fn parse_request(body: &Bytes) -> Result<GuardrailsRequest, FilterError> {
    serde_json::from_slice(body).map_err(|error| request_body_parse_error(&error))
}

/// Build a parse error without including request body content.
fn request_body_parse_error(error: &serde_json::Error) -> FilterError {
    format!(
        "ai_guardrails: could not parse request body at line {} column {}",
        error.line(),
        error.column()
    )
    .into()
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
#[expect(clippy::unwrap_used, reason = "tests")]
mod streaming_tests {
    use bytes::Bytes;

    use super::{parse_request, requests_streaming, stream_requested};

    #[test]
    fn stream_true_is_detected() {
        let body = Bytes::from_static(br#"{"model":"m","messages":[],"stream":true}"#);
        assert!(
            requests_streaming(&body).unwrap(),
            "stream=true should request streaming"
        );
    }

    #[test]
    fn stream_false_is_not_streaming() {
        let body = Bytes::from_static(br#"{"model":"m","messages":[],"stream":false}"#);
        assert!(
            !requests_streaming(&body).unwrap(),
            "stream=false should not request streaming"
        );
    }

    #[test]
    fn absent_stream_is_not_streaming() {
        let body = Bytes::from_static(br#"{"model":"m","messages":[]}"#);
        assert!(
            !requests_streaming(&body).unwrap(),
            "an absent stream field should not request streaming"
        );
    }

    #[test]
    fn malformed_body_fails_closed() {
        let body = Bytes::from_static(b"not json");
        assert!(
            requests_streaming(&body).is_err(),
            "malformed JSON should return an error"
        );
    }

    #[test]
    fn request_parse_errors_do_not_include_body_content() {
        let body = Bytes::from_static(br#""unique-request-secret""#);

        let streaming_error = requests_streaming(&body).unwrap_err();
        let request_error = parse_request(&body).err().unwrap();
        for error in [streaming_error, request_error] {
            let error = error.to_string();
            assert!(
                error.contains("line 1 column"),
                "error should include the parse location: {error}"
            );
            assert!(
                !error.contains("unique-request-secret"),
                "error must not include request body content: {error}"
            );
        }
    }

    #[test]
    fn request_phase_parses_stream_and_messages_together() {
        let body = Bytes::from_static(br#"{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true}"#);
        let request = parse_request(&body).unwrap();

        assert!(
            stream_requested(request.stream.as_ref()),
            "stream=true should request streaming"
        );
        assert_eq!(
            request.into_messages().unwrap().len(),
            1,
            "messages should be available from the same parse"
        );
    }
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
