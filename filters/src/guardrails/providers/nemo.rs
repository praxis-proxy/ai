// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `NeMo` Guardrails provider: calls `/v1/checks` and maps the response
//! to [`GuardResult`].

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method, Uri};
use praxis_ai_apis::subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse};
use praxis_filter::FilterError;
use serde::{Deserialize, Serialize};

use super::{GuardPhase, GuardProvider, GuardResult};

/// Default timeout for `NeMo` HTTP calls (10 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Default maximum number of per-message `/v1/checks` callouts per request.
const DEFAULT_MAX_MESSAGE_CHECKS: u32 = 32;

/// Maximum response body size accepted from `NeMo` (1 MiB).
const MAX_RESPONSE_SIZE: usize = 1024 * 1024;

/// `NeMo`-specific configuration fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoConfig {
    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Allow the endpoint to resolve to non-public addresses.
    #[serde(default)]
    allow_private_endpoint: bool,

    /// Model name sent in each request. Defaults to `""` when omitted.
    #[serde(default)]
    model: String,

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

/// Phase-specific rail selection sent under `guardrails.rail_types`.
#[derive(Serialize)]
struct NemoGuardrailsRequest {
    /// Rail types `NeMo` should run for this check (`input` or `output`).
    rail_types: Vec<&'static str>,
}

/// Outgoing request payload for `NeMo`
#[derive(Serialize)]
struct NemoRequest {
    /// Model name
    model: String,
    /// List of messages to evaluate.
    messages: Vec<serde_json::Value>,
    /// `NeMo` guardrails options for `/v1/checks`.
    guardrails: NemoGuardrailsRequest,
}

/// Incoming response payload for `/v1/checks`.
#[derive(Deserialize)]
struct NemoResponse {
    /// Overall verdict: `"passed"`, `"blocked"`, or `"modified"`.
    status: String,

    /// Content after rails processing.
    #[serde(default)]
    content: String,

    /// Name of the blocking rail, when `status` is `"blocked"`.
    rail: Option<String>,
}

/// `NeMo` Guardrails provider.
pub(in crate::guardrails) struct NemoProvider {
    /// Bounded HTTP client with admission control and circuit breaking.
    client: SubRequestClient,

    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name included in every request. Empty string when not configured.
    model: String,

    /// Per-request deadline covering admission, connect, and I/O.
    timeout: Duration,

    /// Connect-time policy for the configured endpoint.
    address_policy: praxis_ai_apis::callout_target::AddressPolicy,

    /// Maximum per-message `/v1/checks` callouts per proxy request.
    max_message_checks: u32,
}

impl NemoProvider {
    /// Parse and validate `NeMo`-specific config from the provider settings.
    ///
    /// Uses the provided [`SubRequestClient`] so callouts inherit the
    /// runtime's admission control, circuit breaking, and deadline.
    ///
    /// # Errors
    ///
    /// Returns `FilterError` if the configuration is invalid.
    ///
    /// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
    pub fn from_config(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Self, FilterError> {
        let cfg: NemoConfig = serde_yaml::from_value(config.clone())
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): {e}").into() })?;
        if cfg.endpoint.is_empty() {
            return Err("ai_guardrails (nemo): 'endpoint' must not be empty".into());
        }
        let address_policy =
            praxis_ai_apis::callout_target::AddressPolicy::from_allow_private(cfg.allow_private_endpoint);
        praxis_ai_apis::callout_target::validate_configured_http_target(
            "ai_guardrails (nemo)",
            &cfg.endpoint,
            address_policy,
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
            timeout: Duration::from_millis(cfg.timeout_ms),
            address_policy,
            max_message_checks: cfg.max_message_checks,
        })
    }

    /// POST one message slice to `/v1/checks` and map the response.
    async fn check_messages(
        &self,
        messages: Vec<serde_json::Value>,
        phase: GuardPhase,
    ) -> Result<GuardResult, FilterError> {
        let request = build_request(&self.model, messages, phase)?;
        let response = subrequest::execute_url(
            &self.client,
            &self.endpoint,
            request,
            MAX_RESPONSE_SIZE,
            self.timeout,
            self.address_policy,
        )
        .await
        .map_err(|error| map_subrequest_error(&error))?;
        ensure_success_status(&response)?;
        let nemo_response: NemoResponse = serde_json::from_slice(&response.body)
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): failed to parse response: {e}").into() })?;
        map_nemo_response(&nemo_response)
    }
}

#[async_trait]
impl GuardProvider for NemoProvider {
    async fn evaluate(&self, messages: Vec<serde_json::Value>, phase: GuardPhase) -> Result<GuardResult, FilterError> {
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

        let mut pending_redact: Option<GuardResult> = None;
        for end in indices {
            let slice = messages
                .get(..=end)
                .ok_or_else(|| -> FilterError { "ai_guardrails (nemo): invalid message index".into() })?
                .to_vec();
            match apply_slice_result(pending_redact, self.check_messages(slice, phase).await?) {
                Ok(pending) => pending_redact = pending,
                Err(block) => return Ok(block),
            }
        }

        Ok(pending_redact.unwrap_or(GuardResult::Pass))
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
///
/// When `rail_types` is omitted, `NeMo`'s `_determine_rails_from_messages`
/// auto-detects from roles: if both `user` and `assistant` are present it
/// runs **both** input and output rails (`llmrails.py` line 2123-2124).
/// Cumulative request-phase slices include assistant history, so without
/// this override output rails would run on historic assistant text during
/// a request-phase check.
fn rail_types_for_phase(phase: GuardPhase) -> Vec<&'static str> {
    match phase {
        GuardPhase::Request => vec!["input"],
        GuardPhase::Response => vec!["output"],
    }
}

/// Build the outbound `NeMo` JSON callout.
fn build_request(model: &str, messages: Vec<serde_json::Value>, phase: GuardPhase) -> Result<SubRequest, FilterError> {
    let payload = NemoRequest {
        model: model.to_owned(),
        messages,
        guardrails: NemoGuardrailsRequest {
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
        uri: Uri::default(),
        headers,
        body,
    })
}

/// Map a sub-request failure to a filter error, preserving the
/// distinct admission / circuit-open / I/O variants in the message.
fn map_subrequest_error(error: &SubRequestError) -> FilterError {
    format!("ai_guardrails (nemo): failed to send request: {error}").into()
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
/// - `"modified"` - content was transformed (e.g. PII masked)
fn map_nemo_response(nemo: &NemoResponse) -> Result<GuardResult, FilterError> {
    match nemo.status.as_str() {
        "passed" => Ok(GuardResult::Pass),
        "blocked" => Ok(GuardResult::Block {
            reason: nemo.rail.clone().unwrap_or_default(),
        }),
        "modified" => {
            if nemo.content.is_empty() {
                return Err("ai_guardrails (nemo): modified status requires non-empty content".into());
            }
            Ok(GuardResult::Redact {
                modified_text: nemo.content.clone(),
                reason: nemo.rail.clone().unwrap_or_else(|| "modified".to_owned()),
            })
        },
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
    fn target_message_indices_request_collects_user_turns() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "first"}),
            serde_json::json!({"role": "assistant", "content": "reply"}),
            serde_json::json!({"role": "user", "content": "second"}),
        ];

        let indices = target_message_indices(&messages, GuardPhase::Request);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn target_message_indices_response_collects_assistant_turns() {
        let messages = vec![
            serde_json::json!({"role": "assistant", "content": "a"}),
            serde_json::json!({"role": "assistant", "content": "b"}),
        ];

        let indices = target_message_indices(&messages, GuardPhase::Response);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn target_message_indices_empty_when_no_target_role() {
        let messages = vec![serde_json::json!({"role": "system", "content": "sys"})];
        assert!(target_message_indices(&messages, GuardPhase::Request).is_empty());
    }

    #[test]
    fn rail_types_for_request_phase_are_input_only() {
        assert_eq!(rail_types_for_phase(GuardPhase::Request), vec!["input"]);
    }

    #[test]
    fn rail_types_for_response_phase_are_output_only() {
        assert_eq!(rail_types_for_phase(GuardPhase::Response), vec!["output"]);
    }

    #[test]
    fn build_request_includes_phase_specific_rail_types() {
        let request = build_request(
            "test",
            vec![serde_json::json!({"role": "user", "content": "hello"})],
            GuardPhase::Request,
        )
        .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(payload["guardrails"]["rail_types"], serde_json::json!(["input"]));
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

        let err = apply_slice_result(pending, blocked.clone()).unwrap_err();
        assert_eq!(err, blocked);
    }

    #[test]
    fn apply_slice_result_modified_on_later_slice_overwrites_pending_redact() {
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
        let pending = apply_slice_result(pending, second.clone()).unwrap();
        assert_eq!(pending, Some(second));
    }

    #[test]
    fn map_nemo_response_passed_returns_pass() {
        let resp = NemoResponse {
            status: "passed".to_string(),
            content: "hello".to_string(),
            rail: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(matches!(result, GuardResult::Pass));
    }

    #[test]
    fn map_nemo_response_blocked_returns_block_with_rail() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: Some("toxicity".to_string()),
        };
        let result = map_nemo_response(&resp).unwrap();
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
        let result = map_nemo_response(&resp).unwrap();
        assert!(matches!(result, GuardResult::Block { reason } if reason.is_empty()));
    }

    #[test]
    fn map_nemo_response_modified_returns_redact() {
        let resp = NemoResponse {
            status: "modified".to_string(),
            content: "masked text".to_string(),
            rail: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(
            matches!(
                result,
                GuardResult::Redact {
                    modified_text,
                    reason
                } if modified_text == "masked text" && reason == "modified"
            ),
            "modified response should produce GuardResult::Redact"
        );
    }

    #[test]
    fn map_nemo_response_modified_without_content_is_rejected() {
        let resp = NemoResponse {
            status: "modified".to_string(),
            content: String::new(),
            rail: None,
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(
            err_msg.contains("modified status requires non-empty content"),
            "empty modified content should be rejected: {err_msg}"
        );
    }

    #[test]
    fn map_nemo_response_unknown_status_returns_error() {
        let resp = NemoResponse {
            status: "garbage".to_string(),
            content: String::new(),
            rail: None,
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(
            err_msg.contains("unknown status 'garbage'"),
            "unknown status should produce a descriptive error: {err_msg}"
        );
    }
}
