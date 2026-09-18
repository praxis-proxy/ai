// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `NeMo` Guardrails provider: calls `/v1/checks` and maps the response
//! to [`GuardResult`].

use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method};
use praxis_ai_apis::{
    callout_target::{AddressPolicy, validate_configured_http_target, validate_resolved_addrs},
    subrequest::{SubRequest, SubRequestClient, SubResponse},
};
use praxis_core::connectivity::{UrlTargetError, prepare_url_target};
use praxis_filter::{
    CalloutOutcome, CalloutResponse, FilterError, FilteredSubrequestExecutor, RequestExtensions, StagedUpstream,
    StagedUpstreamFallback,
};
use serde::{Deserialize, Serialize};

use super::{GuardCalloutRuntime, GuardPhase, GuardProvider, GuardResult};

/// Default timeout for `NeMo` HTTP calls (10 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Default maximum number of per-message `/v1/checks` callouts per request.
const DEFAULT_MAX_MESSAGE_CHECKS: u32 = 32;

/// Maximum response body size accepted from `NeMo` (1 MiB).
const MAX_RESPONSE_SIZE: usize = 1024 * 1024;

/// Input-only rail selection for request-phase checks.
const INPUT_RAIL_TYPES: &[&str] = &["input"];
/// Output-only rail selection for response-phase checks.
const OUTPUT_RAIL_TYPES: &[&str] = &["output"];

/// `NeMo`-specific configuration fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoConfig {
    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name sent in each request. Defaults to `""` when omitted.
    #[serde(default)]
    model: String,

    /// Optional guardrail configuration selection sent to `NeMo`.
    #[serde(default)]
    guardrails: Option<NemoGuardrails>,

    /// Per-request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,

    /// Maximum number of per-message `/v1/checks` callouts per proxy request.
    #[serde(default = "default_max_message_checks")]
    max_message_checks: u32,
}

/// Returns the default timeout value for serde deserialization.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Returns the default per-request message check limit for serde deserialization.
fn default_max_message_checks() -> u32 {
    DEFAULT_MAX_MESSAGE_CHECKS
}

/// Guardrail configurations selected for evaluation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoGuardrails {
    /// Configuration IDs understood by the configured `NeMo` service.
    config_ids: Vec<String>,
}

/// Phase-specific rail selection sent under `guardrails.rail_types`.
#[derive(Serialize)]
struct NemoGuardrailsRequest<'a> {
    /// Omitted when the service should select its default configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    config_ids: Option<&'a [String]>,
    /// Rail types `NeMo` should run for this check (`input` or `output`).
    rail_types: &'static [&'static str],
}

/// Outgoing request payload for `NeMo`
#[derive(Serialize)]
struct NemoRequest<'a, 'b> {
    /// Model name
    model: &'a str,
    /// List of messages to evaluate.
    messages: &'b [serde_json::Value],
    /// `NeMo` guardrails options for `/v1/checks`.
    guardrails: NemoGuardrailsRequest<'a>,
}

/// Incoming response payload for `/v1/checks`.
#[derive(Deserialize)]
struct NemoResponse {
    /// Overall verdict: `"passed"`, `"blocked"`, or `"modified"`.
    status: String,

    /// Content after rails processing. This field is required by `/v1/checks`;
    /// an empty string remains valid for a full redaction.
    content: String,

    /// Name of the blocking or modifying rail, when supplied.
    rail: Option<String>,
}

/// `NeMo` Guardrails provider.
pub(in crate::guardrails) struct NemoProvider {
    /// Shared HTTP client with admission control and circuit breaking.
    client: SubRequestClient,

    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name included in every request. Empty string when not configured.
    model: String,

    /// Optional guardrail configuration selection.
    guardrails: Option<NemoGuardrails>,

    /// Per-request deadline covering target preparation and the outbound exchange.
    timeout: Duration,

    /// Maximum per-message `/v1/checks` callouts per proxy request.
    max_message_checks: u32,
}

impl NemoProvider {
    /// Parse and validate `NeMo`-specific config from the provider settings.
    ///
    /// Uses the provided [`SubRequestClient`] and bound outbound chain so
    /// callouts inherit the runtime's admission control, circuit breaking,
    /// and outbound security filters.
    ///
    /// # Errors
    ///
    /// Returns `FilterError` if the configuration is invalid.
    pub fn from_config(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Self, FilterError> {
        let cfg: NemoConfig = serde_yaml::from_value(config.clone())
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): {e}").into() })?;
        if cfg.endpoint.is_empty() {
            return Err("ai_guardrails (nemo): 'endpoint' must not be empty".into());
        }
        validate_configured_http_target(
            "ai_guardrails (nemo)",
            &cfg.endpoint,
            // Construction validates URL shape. The bound outbound pipeline's
            // runtime policy is applied after nested pipelines are built and
            // enforced again against the pinned resolution for every callout.
            AddressPolicy::from_allow_private(true),
        )?;
        if cfg.timeout_ms == 0 {
            return Err("ai_guardrails (nemo): 'timeout_ms' must be greater than zero".into());
        }
        if cfg.max_message_checks == 0 {
            return Err("ai_guardrails (nemo): 'max_message_checks' must be greater than zero".into());
        }

        Ok(Self {
            client,
            endpoint: cfg.endpoint,
            model: cfg.model,
            guardrails: cfg.guardrails,
            timeout: Duration::from_millis(cfg.timeout_ms),
            max_message_checks: cfg.max_message_checks,
        })
    }

    /// Return the validated timeout used for the complete provider callout.
    pub(in crate::guardrails) fn callout_timeout(&self) -> Duration {
        self.timeout
    }
}

#[async_trait]
impl GuardProvider for NemoProvider {
    async fn evaluate(
        &self,
        messages: Vec<serde_json::Value>,
        phase: GuardPhase,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<GuardResult, FilterError> {
        let indices = target_message_indices(&messages, phase);
        if indices.is_empty() {
            return Ok(GuardResult::Pass);
        }
        if indices.len() > self.max_message_checks as usize {
            return Err(format!(
                "ai_guardrails (nemo): conversation has {} target messages, exceeding max_message_checks ({})",
                indices.len(),
                self.max_message_checks
            )
            .into());
        }

        let mut pending_redact = None;
        for end in indices {
            runtime
                .deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| -> FilterError {
                    "ai_guardrails (nemo): overall evaluation deadline exceeded".into()
                })?;
            let messages = messages
                .get(..=end)
                .ok_or_else(|| -> FilterError { "ai_guardrails (nemo): invalid message index".into() })?;
            match apply_slice_result(pending_redact, self.check_messages(messages, phase, runtime).await?) {
                Ok(pending) => pending_redact = pending,
                Err(block) => return Ok(block),
            }
        }

        Ok(pending_redact.unwrap_or(GuardResult::Pass))
    }
}

#[expect(
    clippy::multiple_inherent_impl,
    reason = "separates construction from the async callout path"
)]
impl NemoProvider {
    /// POST one message slice to `/v1/checks` through the outbound chain.
    async fn check_messages(
        &self,
        messages: &[serde_json::Value],
        phase: GuardPhase,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<GuardResult, FilterError> {
        let request = build_request(&self.model, messages, phase, self.guardrails.as_ref())?;
        let response = Box::pin(self.execute_callout(request, runtime)).await?;
        ensure_success_status(&response)?;
        let nemo_response: NemoResponse = serde_json::from_slice(&response.body).map_err(|error| -> FilterError {
            format!("ai_guardrails (nemo): failed to parse response: {error}").into()
        })?;
        map_nemo_response(nemo_response)
    }

    /// Execute one `NeMo` request through the bound outbound filter chain.
    #[expect(
        clippy::too_many_lines,
        reason = "target pinning and classified response mapping form one operation"
    )]
    #[expect(
        clippy::large_stack_frames,
        reason = "filtered subrequest executor owns the callout state machine"
    )]
    async fn execute_callout(
        &self,
        request: SubRequest,
        runtime: &GuardCalloutRuntime<'_>,
    ) -> Result<SubResponse, FilterError> {
        let address_policy = AddressPolicy::from_allow_private(runtime.outbound.allow_private_upstreams());
        let target = prepare_url_target(&self.endpoint, runtime.deadline, |addrs| {
            validate_resolved_addrs("ai_guardrails (nemo)", addrs, address_policy)
                .map(|_| ())
                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> { error.to_string().into() })
        })
        .await
        .map_err(map_url_target_error)?;

        let mut extensions = RequestExtensions::default();
        extensions.insert(StagedUpstream::from_prepared_target(&target)?);
        if target.addresses().len() > 1 {
            extensions.insert(StagedUpstreamFallback::from_prepared_target(&target));
        }
        let prepared = target.bind(request);

        let executor = FilteredSubrequestExecutor::for_callout(
            self.client.clone(),
            runtime.downstream.clone(),
            runtime.depth,
            MAX_RESPONSE_SIZE,
            self.timeout,
        );

        match Box::pin(executor.run_classified(runtime.outbound, prepared.request(), extensions, runtime.deadline))
            .await?
        {
            CalloutOutcome::Response(CalloutResponse::Buffered(response)) => Ok(response),
            CalloutOutcome::Response(CalloutResponse::Streaming { .. }) => {
                Err("ai_guardrails (nemo): streaming NeMo responses are not supported".into())
            },
            CalloutOutcome::ResponseTooLarge { actual, limit } => Err(format!(
                "ai_guardrails (nemo): provider response exceeded size limit ({} bytes, limit {limit} bytes)",
                actual.map_or_else(|| "unknown".to_owned(), |size| size.to_string())
            )
            .into()),
            _ => Err("ai_guardrails (nemo): unsupported provider response mode".into()),
        }
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Combine a per-slice verdict into the running evaluation state.
///
/// `blocked` fails fast. `modified` is retained but later slices are still
/// checked so a subsequent `blocked` verdict is not missed.
fn apply_slice_result(
    pending_redact: Option<GuardResult>,
    result: GuardResult,
) -> Result<Option<GuardResult>, GuardResult> {
    match result {
        GuardResult::Pass => Ok(pending_redact),
        block @ GuardResult::Block { .. } => Err(block),
        redact @ GuardResult::Redact { .. } => Ok(Some(redact)),
    }
}

/// Indices of messages whose role matches the active guard phase.
///
/// `/v1/checks` evaluates only the last message of the relevant role per
/// call. To preserve full-conversation coverage, the provider issues one
/// HTTP request per target message, each carrying the prefix of the
/// conversation up to and including that message.
fn target_message_indices(messages: &[serde_json::Value], phase: GuardPhase) -> Vec<usize> {
    let target_role = match phase {
        GuardPhase::Request => "user",
        GuardPhase::Response => "assistant",
    };

    messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.get("role").and_then(|role| role.as_str()) == Some(target_role))
        .map(|(index, _)| index)
        .collect()
}

/// `NeMo` rail types for the active guard phase.
fn rail_types_for_phase(phase: GuardPhase) -> &'static [&'static str] {
    match phase {
        GuardPhase::Request => INPUT_RAIL_TYPES,
        GuardPhase::Response => OUTPUT_RAIL_TYPES,
    }
}

/// Build the outbound `NeMo` JSON callout.
fn build_request<'a>(
    model: &'a str,
    messages: &[serde_json::Value],
    phase: GuardPhase,
    guardrails: Option<&'a NemoGuardrails>,
) -> Result<SubRequest, FilterError> {
    let payload = NemoRequest {
        model,
        messages,
        guardrails: NemoGuardrailsRequest {
            config_ids: guardrails.map(|settings| settings.config_ids.as_slice()),
            rail_types: rail_types_for_phase(phase),
        },
    };
    let body =
        Bytes::from(serde_json::to_vec(&payload).map_err(|e| -> FilterError {
            format!("ai_guardrails (nemo): failed to serialize request: {e}").into()
        })?);

    let mut headers = HeaderMap::new();
    headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(http::header::ACCEPT, HeaderValue::from_static("application/json"));

    Ok(SubRequest {
        method: Method::POST,
        uri: http::Uri::default(),
        headers,
        body,
    })
}

/// Map URL target preparation failures to filter errors.
fn map_url_target_error(error: UrlTargetError) -> FilterError {
    match error {
        UrlTargetError::InvalidTarget(invalid) => {
            format!("ai_guardrails (nemo): invalid endpoint URL: {invalid}").into()
        },
        UrlTargetError::Resolve(resolve) => {
            format!("ai_guardrails (nemo): failed to resolve endpoint: {resolve}").into()
        },
        UrlTargetError::PolicyRejected(source) => {
            format!("ai_guardrails (nemo): endpoint address policy rejected target: {source}").into()
        },
        UrlTargetError::DeadlineExceeded => "ai_guardrails (nemo): endpoint preparation deadline exceeded".into(),
        _ => "ai_guardrails (nemo): endpoint preparation failed".into(),
    }
}

/// Reject non-2xx HTTP responses from the provider.
fn ensure_success_status(response: &SubResponse) -> Result<(), FilterError> {
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "ai_guardrails (nemo): provider returned HTTP status code {}",
            response.status
        )
        .into());
    }
    Ok(())
}

/// Map a deserialized [`NemoResponse`] to a [`GuardResult`].
///
/// The `/v1/checks` endpoint returns three statuses:
/// - `"passed"` - all rails passed
/// - `"blocked"` - at least one rail blocked the content
/// - `"modified"` - content was transformed (for example, PII masking)
fn map_nemo_response(nemo: NemoResponse) -> Result<GuardResult, FilterError> {
    let NemoResponse { status, content, rail } = nemo;
    match status.as_str() {
        "passed" => Ok(GuardResult::Pass),
        "blocked" => Ok(GuardResult::Block {
            reason: rail.unwrap_or_default(),
        }),
        "modified" => Ok(GuardResult::Redact {
            modified_text: content,
            reason: rail.unwrap_or_else(|| "modified".to_owned()),
        }),
        other => Err(format!("ai_guardrails (nemo): unknown status '{other}'").into()),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::str_to_string,
    clippy::indexing_slicing,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn request_serializes_configured_guardrails() {
        let config: NemoConfig = serde_yaml::from_str(
            r#"
endpoint: "http://localhost:8000/v1/checks"
model: check-model
guardrails:
  config_ids: [your-config, another-config]
"#,
        )
        .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let request = build_request(
            &config.model,
            &messages,
            GuardPhase::Request,
            config.guardrails.as_ref(),
        )
        .unwrap();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            serde_json::json!({
                "model": "check-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "guardrails": {"rail_types": ["input"], "config_ids": ["your-config", "another-config"]}
            })
        );
    }

    #[test]
    fn request_omits_unconfigured_config_ids() {
        let config: NemoConfig = serde_yaml::from_str("endpoint: http://localhost:8000").unwrap();
        let request = build_request(&config.model, &[], GuardPhase::Response, config.guardrails.as_ref()).unwrap();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            serde_json::json!({
                "model": "", "messages": [], "guardrails": {"rail_types": ["output"]}
            })
        );
    }

    #[test]
    fn target_message_indices_request_collects_user_turns() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "first"}),
            serde_json::json!({"role": "assistant", "content": "reply"}),
            serde_json::json!({"role": "user", "content": "second"}),
        ];

        assert_eq!(target_message_indices(&messages, GuardPhase::Request), vec![1, 3]);
    }

    #[test]
    fn target_message_indices_response_collects_assistant_turns() {
        let messages = vec![
            serde_json::json!({"role": "assistant", "content": "a"}),
            serde_json::json!({"role": "assistant", "content": "b"}),
        ];

        assert_eq!(target_message_indices(&messages, GuardPhase::Response), vec![0, 1]);
    }

    #[test]
    fn target_message_indices_empty_when_no_target_role() {
        let messages = vec![serde_json::json!({"role": "system", "content": "sys"})];
        assert!(target_message_indices(&messages, GuardPhase::Request).is_empty());
    }

    #[test]
    fn rail_types_are_phase_specific() {
        assert_eq!(rail_types_for_phase(GuardPhase::Request), ["input"]);
        assert_eq!(rail_types_for_phase(GuardPhase::Response), ["output"]);
    }

    #[test]
    fn build_request_includes_phase_specific_rail_types() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let request = build_request("test", &messages, GuardPhase::Request, None).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&request.body).unwrap();

        assert_eq!(payload["guardrails"]["rail_types"], serde_json::json!(["input"]));
        assert_eq!(payload["messages"], serde_json::Value::Array(messages));
    }

    #[test]
    fn apply_slice_result_modified_then_blocked_fails_fast_on_blocked() {
        let modified = GuardResult::Redact {
            modified_text: "masked".into(),
            reason: "pii".into(),
        };
        let blocked = GuardResult::Block {
            reason: "toxicity".into(),
        };

        let pending = apply_slice_result(None, modified.clone()).unwrap();
        assert_eq!(pending, Some(modified));
        assert_eq!(apply_slice_result(pending, blocked.clone()).unwrap_err(), blocked);
    }

    #[test]
    fn apply_slice_result_keeps_latest_modified_verdict() {
        let first = GuardResult::Redact {
            modified_text: "first".into(),
            reason: "pii".into(),
        };
        let second = GuardResult::Redact {
            modified_text: "second".into(),
            reason: "pii".into(),
        };

        let pending = apply_slice_result(None, first).unwrap();
        let pending = apply_slice_result(pending, GuardResult::Pass).unwrap();
        assert_eq!(apply_slice_result(pending, second.clone()).unwrap(), Some(second));
    }

    #[test]
    fn map_nemo_response_passed_returns_pass() {
        let resp = NemoResponse {
            status: "passed".to_string(),
            content: "hello".to_string(),
            rail: None,
        };
        let result = map_nemo_response(resp).unwrap();
        assert!(matches!(result, GuardResult::Pass));
    }

    #[test]
    fn map_nemo_response_blocked_returns_block_with_rail() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: Some("toxicity".to_string()),
        };
        let result = map_nemo_response(resp).unwrap();
        assert!(
            matches!(result, GuardResult::Block { reason } if reason == "toxicity"),
            "blocked response should produce GuardResult::Block with rail name as reason"
        );
    }

    #[test]
    fn map_nemo_response_blocked_without_rail_returns_empty_reason() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: None,
        };
        let result = map_nemo_response(resp).unwrap();
        assert!(matches!(result, GuardResult::Block { reason } if reason.is_empty()));
    }

    #[test]
    fn map_nemo_response_modified_returns_redact() {
        let resp = NemoResponse {
            status: "modified".to_string(),
            content: "masked text".to_string(),
            rail: Some("pii".to_string()),
        };
        assert_eq!(
            map_nemo_response(resp).unwrap(),
            GuardResult::Redact {
                modified_text: "masked text".to_string(),
                reason: "pii".to_string(),
            }
        );
    }

    #[test]
    fn deserialize_response_requires_content_but_allows_empty_content() {
        let missing = serde_json::from_value::<NemoResponse>(serde_json::json!({"status": "passed"}));
        assert!(missing.is_err(), "missing /v1/checks content must fail closed");

        let empty = serde_json::from_value::<NemoResponse>(serde_json::json!({
            "status": "modified",
            "content": ""
        }))
        .unwrap();
        assert!(empty.content.is_empty());
    }

    #[test]
    fn map_nemo_response_unknown_status_returns_error() {
        let resp = NemoResponse {
            status: "garbage".to_string(),
            content: String::new(),
            rail: None,
        };
        let err_msg = format!("{}", map_nemo_response(resp).unwrap_err());
        assert!(
            err_msg.contains("unknown status 'garbage'"),
            "unknown status should produce a descriptive error: {err_msg}"
        );
    }
}
