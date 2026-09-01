// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `NeMo` Guardrails provider: calls `/v1/guardrail/checks` and maps
//! the response to [`GuardResult`].

use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt as _;
use praxis_filter::FilterError;
use serde::{Deserialize, Serialize};

use super::{GuardPhase, GuardProvider, GuardResult};

/// Default timeout for `NeMo` HTTP calls (10 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Maximum response body size accepted from `NeMo` (1 MiB).
const MAX_RESPONSE_SIZE: usize = 1024 * 1024;

/// `NeMo`-specific configuration fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoConfig {
    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name sent in each request. Defaults to `""` when omitted.
    #[serde(default)]
    model: String,

    /// Per-request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

/// Returns the default timeout value for serde deserialization.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Outgoing request payload for `NeMo`
#[derive(Serialize)]
struct NemoRequest {
    /// Model name
    model: String,
    /// List of messages to evaluate.
    messages: Vec<serde_json::Value>,
}

/// Incoming response payload for `NeMo`
#[derive(Deserialize)]
struct NemoResponse {
    /// Overall verdict: `"success"`, `"blocked"`, or `"error"`.
    status: String,

    /// Per-rail evaluation results. The names of rails whose `status` is
    /// `"blocked"` are joined to form the [`GuardResult::Block::reason`]
    /// string.
    rails_status: Option<serde_json::Value>,

    /// Error details returned when `status` is `"error"`.
    /// Contains `"error"` and optionally `"details"` keys.
    guardrails_data: Option<serde_json::Value>,
}

/// `NeMo` Guardrails provider.
pub(in crate::guardrails) struct NemoProvider {
    /// Pre-configured HTTP client.
    client: reqwest::Client,

    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name included in every request. Empty string when not configured.
    model: String,
}

impl NemoProvider {
    /// Parse and validate `NeMo`-specific config from the provider settings.
    ///
    /// Builds a new `NeMo` provider with a pre-configured HTTP client.
    ///
    /// # Errors
    ///
    /// Returns `FilterError` if the configuration is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg: NemoConfig = serde_yaml::from_value(config.clone())
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): {e}").into() })?;
        if cfg.endpoint.is_empty() {
            return Err("ai_guardrails (nemo): 'endpoint' must not be empty".into());
        }
        if cfg.timeout_ms == 0 {
            return Err("ai_guardrails (nemo): 'timeout_ms' must be greater than zero".into());
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.timeout_ms))
            .build()
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): failed to build HTTP client: {e}").into() })?;

        Ok(Self {
            client,
            endpoint: cfg.endpoint,
            model: cfg.model,
        })
    }
}

#[async_trait]
impl GuardProvider for NemoProvider {
    async fn evaluate(&self, messages: Vec<serde_json::Value>, _phase: GuardPhase) -> Result<GuardResult, FilterError> {
        let payload = NemoRequest {
            model: self.model.clone(),
            messages,
        };
        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): failed to send request: {e}").into() })?;
        check_content_length(&response)?;
        ensure_success_status(&response)?;
        let response_body = read_response_body(response).await?;
        let nemo_response: NemoResponse = serde_json::from_slice(&response_body)
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): failed to parse response: {e}").into() })?;
        map_nemo_response(&nemo_response)
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Reject responses whose declared `Content-Length` exceeds [`MAX_RESPONSE_SIZE`].
fn check_content_length(response: &reqwest::Response) -> Result<(), FilterError> {
    let Some(len) = response.content_length() else {
        return Ok(());
    };
    if usize::try_from(len).map_or(true, |l| l > MAX_RESPONSE_SIZE) {
        return Err(format!(
            "ai_guardrails (nemo): response Content-Length too large \
             ({len} bytes, limit {MAX_RESPONSE_SIZE})"
        )
        .into());
    }
    Ok(())
}

/// Reject non-2xx HTTP responses from the provider.
fn ensure_success_status(response: &reqwest::Response) -> Result<(), FilterError> {
    let status = response.status();
    if !status.is_success() {
        return Err(format!("ai_guardrails (nemo): provider returned HTTP status code {status}").into());
    }
    Ok(())
}

/// Read the response body incrementally, aborting as soon as the
/// running total exceeds [`MAX_RESPONSE_SIZE`].
async fn read_response_body(response: reqwest::Response) -> Result<Bytes, FilterError> {
    let mut stream = response.bytes_stream();
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| -> FilterError {
            format!("ai_guardrails (nemo): failed to read response body: {e}").into()
        })?;
        if body.len() + chunk.len() > MAX_RESPONSE_SIZE {
            return Err(format!(
                "ai_guardrails (nemo): response body too large \
                 (limit {MAX_RESPONSE_SIZE} bytes)"
            )
            .into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

/// Map a deserialized [`NemoResponse`] to a [`GuardResult`].
///
/// The `/v1/guardrail/checks` endpoint returns three statuses:
/// - `"success"` - all rails passed
/// - `"blocked"` - at least one rail blocked the content
/// - `"error"` - `NeMo` internal processing error (fail-closed)
fn map_nemo_response(nemo: &NemoResponse) -> Result<GuardResult, FilterError> {
    match nemo.status.as_str() {
        "success" => Ok(GuardResult::Pass),
        "blocked" => {
            let reason = blocked_rail_names(nemo.rails_status.as_ref());
            Ok(GuardResult::Block { reason })
        },
        "error" => {
            let detail = extract_error_detail(nemo.guardrails_data.as_ref());
            Err(format!("ai_guardrails (nemo): NeMo returned error status: {detail}").into())
        },
        other => Err(format!("ai_guardrails (nemo): unknown status '{other}'").into()),
    }
}

/// Collect the names of all rails whose `status` is `"blocked"` from the
/// `rails_status` map and join them with `", "` in sorted order.
///
/// Returns an empty string if `rails_status` is absent or no rails are blocked.
fn blocked_rail_names(rails_status: Option<&serde_json::Value>) -> String {
    let Some(map) = rails_status.and_then(|v| v.as_object()) else {
        return String::new();
    };
    let mut names: Vec<&str> = map
        .iter()
        .filter(|(_, v)| v.get("status").and_then(|s| s.as_str()) == Some("blocked"))
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    names.join(", ")
}

/// Extract a human-readable error string from the `guardrails_data` object
/// returned by `NeMo` when `status` is `"error"`.
///
/// Expected shape: `{"error": "...", "details": "..."}`.
fn extract_error_detail(data: Option<&serde_json::Value>) -> String {
    let Some(obj) = data.and_then(|v| v.as_object()) else {
        return "no details available".to_owned();
    };
    let error = obj.get("error").and_then(|v| v.as_str()).unwrap_or("unknown error");
    match obj.get("details").and_then(|v| v.as_str()) {
        Some(details) => format!("{error} ({details})"),
        None => error.to_owned(),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::str_to_string, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn blocked_rail_names_sorts_alphabetically() {
        let rails = serde_json::json!({
            "toxicity": {"status": "blocked"},
            "jailbreak": {"status": "blocked"},
            "pii masking": {"status": "blocked"},
        });
        assert_eq!(blocked_rail_names(Some(&rails)), "jailbreak, pii masking, toxicity");
    }

    #[test]
    fn blocked_rail_names_filters_out_non_blocked_rails() {
        let rails = serde_json::json!({
            "toxicity": {"status": "blocked"},
            "jailbreak": {"status": "success"},
        });
        assert_eq!(
            blocked_rail_names(Some(&rails)),
            "toxicity",
            "only rails with status 'blocked' should be included in the reason string"
        );
    }

    #[test]
    fn blocked_rail_names_empty_rails_status_returns_empty_string() {
        let rails = serde_json::json!({});
        assert_eq!(blocked_rail_names(Some(&rails)), "");
    }

    #[test]
    fn blocked_rail_names_absent_map_returns_empty_string() {
        assert_eq!(
            blocked_rail_names(None),
            "",
            "missing rails_status should not panic or error"
        );
    }

    #[test]
    fn blocked_rail_names_non_object_rails_status_returns_empty_string() {
        let rails = serde_json::json!("not an object");
        assert_eq!(blocked_rail_names(Some(&rails)), "");
    }

    #[test]
    fn map_nemo_response_success_returns_pass() {
        let resp = NemoResponse {
            status: "success".to_string(),
            rails_status: None,
            guardrails_data: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(matches!(result, GuardResult::Pass));
    }

    #[test]
    fn map_nemo_response_blocked_returns_block_with_reason() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            rails_status: Some(serde_json::json!({
                "toxicity": {"status": "blocked"},
            })),
            guardrails_data: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(
            matches!(result, GuardResult::Block { reason } if reason == "toxicity"),
            "blocked response should produce GuardResult::Block with rail name as reason"
        );
    }

    #[test]
    fn map_nemo_response_error_extracts_guardrails_data() {
        let resp = NemoResponse {
            status: "error".to_string(),
            rails_status: None,
            guardrails_data: Some(serde_json::json!({
                "error": "Could not load guardrails configuration.",
                "details": "Invalid config path /app/config/nonexistent-config."
            })),
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(
            err_msg.contains("Could not load guardrails configuration."),
            "{err_msg}"
        );
        assert!(err_msg.contains("Invalid config path"), "{err_msg}");
    }

    #[test]
    fn map_nemo_response_error_without_guardrails_data() {
        let resp = NemoResponse {
            status: "error".to_string(),
            rails_status: None,
            guardrails_data: None,
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(err_msg.contains("no details available"), "{err_msg}");
    }

    #[test]
    fn map_nemo_response_error_with_error_key_only() {
        let resp = NemoResponse {
            status: "error".to_string(),
            rails_status: None,
            guardrails_data: Some(serde_json::json!({"error": "Internal failure"})),
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(err_msg.contains("Internal failure"), "{err_msg}");
        assert!(
            !err_msg.contains("Internal failure ("),
            "should not append details in parens when details key is absent: {err_msg}"
        );
    }

    #[test]
    fn map_nemo_response_error_with_missing_error_key() {
        let resp = NemoResponse {
            status: "error".to_string(),
            rails_status: None,
            guardrails_data: Some(serde_json::json!({"unexpected": "shape"})),
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(err_msg.contains("unknown error"), "{err_msg}");
    }

    #[test]
    fn map_nemo_response_unknown_status_returns_error() {
        let resp = NemoResponse {
            status: "garbage".to_string(),
            rails_status: None,
            guardrails_data: None,
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(
            err_msg.contains("unknown status 'garbage'"),
            "unknown status should produce a descriptive error: {err_msg}"
        );
    }
}
