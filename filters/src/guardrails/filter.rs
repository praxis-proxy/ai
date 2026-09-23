// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`AiGuardrailsFilter`] implementation and `HttpFilter` trait impl.

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use bytes::Bytes;
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
    providers::{GuardCalloutRuntime, GuardPhase, GuardProvider, GuardResult, nemo},
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
/// **Wire format:** Chat Completions only (`messages` on requests,
/// `choices[].message` on responses). Responses API, Anthropic Messages,
/// and MCP are not supported yet (see ai#1043).
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
/// ```
pub struct AiGuardrailsFilter {
    /// Guard provider instance.
    provider: Box<dyn GuardProvider>,
    /// Which phases to evaluate.
    phase: PhaseConfig,
    /// Prebuilt outbound filter chain for provider callouts.
    outbound: Arc<FilterPipeline>,
    /// Per-callout deadline derived from the provider configuration.
    callout_timeout: std::time::Duration,
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
        let (provider, callout_timeout): (Box<dyn GuardProvider>, _) = match config.provider.provider_type {
            ProviderType::Nemo => {
                let provider = nemo::NemoProvider::from_config(&config.provider.config, client)?;
                let timeout = provider.callout_timeout();
                (Box::new(provider), timeout)
            },
        };

        Ok(Box::new(Self {
            provider,
            phase: config.phase,
            outbound,
            callout_timeout,
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
        let runtime = self.callout_runtime(ctx);
        let result = self.provider.evaluate(messages, GuardPhase::Request, &runtime).await?;
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
            tokio::task::block_in_place(|| {
                handle.block_on(self.provider.evaluate(messages, GuardPhase::Response, &runtime))
            })
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
            tracing::warn!(verdict, phase = phase_label, %reason, "ai_guardrails: verdict; forwarding unchanged until #49");
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
