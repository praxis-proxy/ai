// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Fail-closed, pre-forward token ceilings.
//!
//! This filter intentionally does not share the admission state machine used
//! by `token_rate_limit`: a ceiling is a static request policy and must remain
//! enforced when a quota backend is unavailable. Place it after request
//! translation/enrichment so it evaluates the provider-bound body.

#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    reason = "private configuration details are covered by the public filter contract"
)]

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use serde::Deserialize;

const DEFAULT_MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenCeilingConfig {
    #[serde(default)]
    max_input_tokens: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    tokenizer: Tokenizer,
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Tokenizer {
    #[default]
    Cl100kBase,
    O200kBase,
}

impl Tokenizer {
    fn count(self, text: &str) -> usize {
        match self {
            Self::Cl100kBase => tiktoken_rs::cl100k_base_singleton().count_ordinary(text),
            Self::O200kBase => tiktoken_rs::o200k_base_singleton().count_ordinary(text),
        }
    }
}

/// Rejects provider-bound requests whose estimated prompt or requested output
/// exceeds configured limits. Missing output-limit fields are rejected when
/// `max_output_tokens` is configured; this strict behavior prevents providers
/// from applying an unbounded model default.
pub struct TokenCeilingFilter {
    max_input_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    tokenizer: Tokenizer,
    max_body_bytes: usize,
}

impl TokenCeilingFilter {
    /// Build a token ceiling filter from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is malformed or does not
    /// define a positive input/output token limit or body-size limit.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenCeilingConfig = parse_filter_config("token_ceiling", config)?;
        if cfg.max_input_tokens.is_none() && cfg.max_output_tokens.is_none() {
            return Err("token_ceiling: at least one of max_input_tokens or max_output_tokens is required".into());
        }
        if cfg.max_input_tokens == Some(0) || cfg.max_output_tokens == Some(0) {
            return Err("token_ceiling: token limits must be greater than zero".into());
        }
        if cfg.max_body_bytes == 0 {
            return Err("token_ceiling: max_body_bytes must be greater than zero".into());
        }
        Ok(Box::new(Self {
            max_input_tokens: cfg.max_input_tokens,
            max_output_tokens: cfg.max_output_tokens,
            tokenizer: cfg.tokenizer,
            max_body_bytes: cfg.max_body_bytes,
        }))
    }

    fn rejection(code: &'static str, message: impl Into<String>, status: u16) -> FilterAction {
        let body = serde_json::json!({
            "error": {
                "type": "token_ceiling_exceeded",
                "message": message.into(),
                "code": code,
            }
        });
        let body =
            serde_json::to_vec(&body).unwrap_or_else(|_| b"{\"error\":{\"type\":\"token_ceiling_exceeded\"}}".to_vec());
        FilterAction::Reject(
            Rejection::status(status)
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(body)),
        )
    }
}

#[async_trait]
impl HttpFilter for TokenCeilingFilter {
    fn name(&self) -> &'static str {
        "token_ceiling"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    #[expect(clippy::too_many_lines, reason = "linear validation with early rejection branches")]
    async fn on_request_body(
        &self,
        _ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(raw) = body.as_ref() else {
            return Ok(Self::rejection(
                "missing_request_body",
                "token ceiling requires a JSON request body".to_owned(),
                400,
            ));
        };
        let value: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(value) => value,
            Err(_) => {
                return Ok(Self::rejection(
                    "invalid_json",
                    "token ceiling requires a valid JSON request body".to_owned(),
                    400,
                ));
            },
        };

        if let Some(limit) = self.max_output_tokens {
            let requested = value
                .get("max_output_tokens")
                .or_else(|| value.get("max_completion_tokens"))
                .or_else(|| value.get("max_tokens"));
            let Some(requested) = requested.and_then(serde_json::Value::as_u64) else {
                return Ok(Self::rejection(
                    "missing_output_limit",
                    format!("request must specify an output token limit no greater than {limit}"),
                    400,
                ));
            };
            if requested > limit {
                return Ok(Self::rejection(
                    "max_output_tokens_exceeded",
                    format!("requested output tokens ({requested}) exceed maximum allowed ({limit})"),
                    400,
                ));
            }
        }

        if let Some(limit) = self.max_input_tokens {
            let serialized = serde_json::to_string(&value).map_err(|error| -> FilterError {
                format!("token_ceiling: failed to serialize request: {error}").into()
            })?;
            let estimated = self.tokenizer.count(&serialized) as u64;
            if estimated > limit {
                return Ok(Self::rejection(
                    "max_input_tokens_exceeded",
                    format!("estimated input tokens ({estimated}) exceed maximum allowed ({limit})"),
                    400,
                ));
            }
        }

        Ok(FilterAction::Continue)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use http::Method;
    use praxis_filter::HttpFilter;

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    fn filter(yaml: &str) -> Box<dyn HttpFilter> {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        TokenCeilingFilter::from_config(&value).unwrap()
    }

    #[tokio::test]
    async fn rejects_missing_output_limit() {
        let filter = filter("max_output_tokens: 10");
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::from_static(br#"{"messages":[]}"#));
        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Reject(_)));
    }

    #[tokio::test]
    async fn rejects_requested_output_above_limit() {
        let filter = filter("max_output_tokens: 10");
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::from_static(br#"{"max_tokens":11}"#));
        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Reject(_)));
    }

    #[tokio::test]
    async fn accepts_request_within_both_limits() {
        let filter = filter("max_input_tokens: 100\nmax_output_tokens: 10");
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
        ));
        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }
}
