// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Fail-closed, pre-forward token ceilings.
//!
//! This filter intentionally does not share the admission state machine used
//! by `token_rate_limit`: a ceiling is a static request policy and must remain
//! enforced when a quota backend is unavailable. Place it after request
//! translation/enrichment so it evaluates the provider-bound body.

#![cfg_attr(
    not(test),
    expect(
        clippy::missing_docs_in_private_items,
        reason = "private configuration details are covered by the public filter contract"
    )
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
    /// Maximum number of tokens in the serialized provider-bound JSON body.
    /// This includes JSON keys, model/tool schemas, and non-text data.
    #[serde(default)]
    max_input_tokens: Option<u64>,
    /// Maximum requested output tokens. The request must contain exactly one
    /// of `max_output_tokens`, `max_completion_tokens`, or `max_tokens`.
    #[serde(default)]
    max_output_tokens: Option<u64>,
    /// Tiktoken encoding used for the serialized request-body estimate.
    #[serde(default)]
    tokenizer: Tokenizer,
    /// Maximum request-body bytes buffered for validation. Defaults to 2 MiB.
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
    /// Eagerly load the selected BPE vocabulary during pipeline construction.
    fn initialize(self) {
        match self {
            Self::Cl100kBase => {
                let _ = tiktoken_rs::cl100k_base_singleton();
            },
            Self::O200kBase => {
                let _ = tiktoken_rs::o200k_base_singleton();
            },
        }
    }

    fn count(self, text: &str) -> usize {
        match self {
            Self::Cl100kBase => tiktoken_rs::cl100k_base_singleton().count_ordinary(text),
            Self::O200kBase => tiktoken_rs::o200k_base_singleton().count_ordinary(text),
        }
    }
}

/// Rejects provider-bound requests whose estimated serialized request body or
/// requested output exceeds configured limits. Missing output-limit fields are
/// rejected when `max_output_tokens` is configured; this strict behavior
/// prevents providers from applying an unbounded model default. At least one
/// of `max_input_tokens` or `max_output_tokens` is required, and
/// `max_body_bytes` defaults to 2 MiB.
///
/// # YAML
///
/// ```yaml
/// filter: token_ceiling
/// max_input_tokens: 4000
/// max_output_tokens: 1024
/// tokenizer: cl100k_base
/// max_body_bytes: 2097152
/// ```
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
        cfg.tokenizer.initialize();
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

        let Some(raw) = body.as_ref().filter(|raw| !raw.is_empty()) else {
            return Ok(FilterAction::Continue);
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
            let present = ["max_output_tokens", "max_completion_tokens", "max_tokens"]
                .into_iter()
                .filter_map(|name| value.get(name).map(|value| (name, value)))
                .collect::<Vec<_>>();
            if present.len() > 1 {
                let fields = present.iter().map(|(name, _)| *name).collect::<Vec<_>>().join(", ");
                return Ok(Self::rejection(
                    "multiple_output_limits",
                    format!("request must specify only one output token limit; found {fields}"),
                    400,
                ));
            }
            let Some((field, requested)) = present.into_iter().next() else {
                return Ok(Self::rejection(
                    "missing_output_limit",
                    format!("request must specify an output token limit no greater than {limit}"),
                    400,
                ));
            };
            let Some(requested) = requested.as_u64() else {
                return Ok(Self::rejection(
                    "invalid_output_limit",
                    format!("{field} must be a non-negative integer no greater than {limit}"),
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
            // Every token occupies at least one UTF-8 byte, so a body no
            // larger than the token ceiling cannot exceed it after encoding.
            let len = raw.len() as u64;
            let estimated = if len <= limit {
                len
            } else {
                let tokenizer = self.tokenizer;
                let raw = raw.clone();
                tokio::task::spawn_blocking(move || {
                    std::str::from_utf8(&raw).map_or(u64::MAX, |text| tokenizer.count(text) as u64)
                })
                .await
                .map_err(|error| -> FilterError {
                    format!("token_ceiling: tokenization task failed: {error}").into()
                })?
            };
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
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests use unwrap/panic for fixture failures"
)]
mod tests {
    use http::Method;
    use praxis_filter::HttpFilter;

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    fn filter(yaml: &str) -> Box<dyn HttpFilter> {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        TokenCeilingFilter::from_config(&value).unwrap()
    }

    async fn action(yaml: &str, body: Option<Bytes>) -> FilterAction {
        let filter = filter(yaml);
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        let mut body = body;
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap()
    }

    #[tokio::test]
    async fn rejects_missing_output_limit() {
        let action = action("max_output_tokens: 10", Some(Bytes::from_static(br#"{"messages":[]}"#))).await;
        assert!(matches!(action, FilterAction::Reject(_)));
    }

    #[tokio::test]
    async fn rejects_requested_output_above_limit() {
        let action = action(
            "max_output_tokens: 10",
            Some(Bytes::from_static(br#"{"max_tokens":11}"#)),
        )
        .await;
        assert!(matches!(action, FilterAction::Reject(_)));
    }

    #[tokio::test]
    async fn accepts_request_within_both_limits() {
        let action = action(
            "max_input_tokens: 100\nmax_output_tokens: 10",
            Some(Bytes::from_static(
                br#"{"max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
            )),
        )
        .await;
        assert!(matches!(action, FilterAction::Continue));
    }

    #[tokio::test]
    async fn bodyless_requests_are_not_applicable() {
        for body in [None, Some(Bytes::new())] {
            let action = action("max_input_tokens: 10", body).await;
            assert!(matches!(action, FilterAction::Continue));
        }
    }

    #[tokio::test]
    async fn rejects_serialized_request_over_input_limit() {
        let action = action("max_input_tokens: 1", Some(Bytes::from_static(br#"{"messages":[]}"#))).await;
        assert!(matches!(action, FilterAction::Reject(_)));
    }

    #[tokio::test]
    async fn accepts_each_supported_output_limit_field() {
        for field in ["max_output_tokens", "max_completion_tokens", "max_tokens"] {
            let body = Bytes::from(format!(r#"{{"{field}":10}}"#));
            let action = action("max_output_tokens: 10", Some(body)).await;
            assert!(matches!(action, FilterAction::Continue), "field={field}");
        }
    }

    #[tokio::test]
    async fn rejects_multiple_output_limit_fields() {
        let action = action(
            "max_output_tokens: 10",
            Some(Bytes::from_static(br#"{"max_tokens":10,"max_output_tokens":1}"#)),
        )
        .await;
        assert!(matches!(action, FilterAction::Reject(_)));
    }

    #[tokio::test]
    async fn accepts_o200k_tokenizer() {
        let action = action(
            "max_input_tokens: 10\ntokenizer: o200k_base",
            Some(Bytes::from_static(br#"{"input":"hello world"}"#)),
        )
        .await;
        assert!(matches!(action, FilterAction::Continue));
    }

    #[test]
    fn rejects_invalid_configuration() {
        for (yaml, expected) in [
            ("{}", "at least one"),
            ("max_input_tokens: 0", "token limits"),
            ("max_output_tokens: 0", "token limits"),
            ("max_input_tokens: 1\nmax_body_bytes: 0", "max_body_bytes"),
            ("max_input_tokens: 1\nunknown: true", "unknown field"),
        ] {
            let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
            let Err(error) = TokenCeilingFilter::from_config(&value) else {
                panic!("configuration should be rejected: {yaml}");
            };
            assert!(error.to_string().contains(expected), "yaml={yaml}");
        }
    }
}
