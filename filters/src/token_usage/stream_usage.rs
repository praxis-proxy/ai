// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Injects `stream_options.include_usage` into OpenAI streaming requests.
//!
//! OpenAI Chat Completions only reports token usage in a stream when the
//! client sends `stream_options: {"include_usage": true}`. Without it the
//! response carries no usage object and the gateway meters a legitimate 0.
//!
//! This filter ensures every streaming chat-completions request carries the
//! opt-in, so metering works regardless of what the client sends.
//!
//! Non-streaming requests and bodies that already carry the opt-in pass
//! through untouched.
//!
//! The filter buffers the request body up to `max_body_bytes`, and that cap
//! applies to every request traversing the chain — not only streaming chat
//! completions. The default matches the workspace-wide 10 MiB JSON body
//! default used by the other body-rewriting filters; deployments with larger
//! legitimate requests should raise `max_body_bytes` (ceiling 64 MiB).
//!
//! # YAML
//!
//! ```yaml
//! filter: stream_usage_inject
//! ```

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::{json_body::replace_json_body, operation::normalize_path};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::Deserialize;
use serde_json::Value;
use tracing::debug;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Whether the request targets a Chat Completions endpoint.
///
/// Only Chat Completions reports usage through `stream_options.include_usage`;
/// the Responses API accepts `stream_options` but has no `include_usage` field
/// (it carries `include_obfuscation` instead), so injecting there would add a
/// field the provider rejects. The path is normalized with the shared
/// operation-registry policy — query string ignored, one trailing slash
/// tolerated — so `POST /v1/chat/completions/` cannot bypass injection.
fn is_chat_completions(ctx: &HttpFilterContext<'_>) -> bool {
    normalize_path(ctx.request.uri.path()).ends_with("/chat/completions")
}

/// Returns `true` when the body is a streaming request without `include_usage`.
fn needs_injection(value: &Value) -> bool {
    value.get("stream") == Some(&Value::Bool(true))
        && value.get("stream_options").and_then(|so| so.get("include_usage")) != Some(&Value::Bool(true))
}

/// Sets `stream_options.include_usage = true`, creating the object if needed.
///
/// Returns whether the body was modified. A `stream_options` value that is
/// present but is neither an object nor `null` is left untouched and reported
/// as unmodified: the request is provider-invalid input, and the gateway
/// should pass it through for the upstream to reject rather than rewriting a
/// field into a shape the client never sent.
fn inject_include_usage(value: &mut Value) -> bool {
    let Some(obj) = value.as_object_mut() else {
        return false;
    };

    match obj.get_mut("stream_options") {
        Some(Value::Object(options)) => {
            options.insert("include_usage".to_owned(), Value::Bool(true));
        },
        Some(Value::Null) | None => {
            let mut options = serde_json::Map::new();
            options.insert("include_usage".to_owned(), Value::Bool(true));
            obj.insert("stream_options".to_owned(), Value::Object(options));
        },
        Some(present) => {
            debug!(
                "stream_options present but not an object ({present}); leaving request untouched for upstream \
                 validation"
            );
            return false;
        },
    }

    debug!("injected stream_options.include_usage=true");
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// Deserialized YAML config for `stream_usage_inject`.
struct StreamUsageConfig {
    /// Maximum request body bytes for `StreamBuffer` mode. The cap applies to
    /// every request traversing the chain, not only streaming chat
    /// completions.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// Returns the default max body bytes (workspace-wide 10 MiB JSON body
/// default).
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Injects `stream_options.include_usage = true` into streaming OpenAI
/// chat-completions requests so the upstream response contains token usage.
pub struct StreamUsageInjectFilter {
    /// Maximum request body bytes for `StreamBuffer` mode.
    max_body_bytes: usize,
}

impl StreamUsageInjectFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or
    /// `max_body_bytes` is zero or exceeds the 64 MiB ceiling.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: StreamUsageConfig = parse_filter_config("stream_usage_inject", config)?;
        validate_max_body_bytes("stream_usage_inject", cfg.max_body_bytes)?;
        Ok(Box::new(Self {
            max_body_bytes: cfg.max_body_bytes,
        }))
    }
}

#[async_trait]
impl HttpFilter for StreamUsageInjectFilter {
    fn name(&self) -> &'static str {
        "stream_usage_inject"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
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

        if !is_chat_completions(ctx) {
            return Ok(FilterAction::Continue);
        }

        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        let mut value: Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(err) => {
                debug!("chat-completions request body is not valid JSON ({err}); passing through untouched");
                return Ok(FilterAction::Continue);
            },
        };

        if needs_injection(&value) && inject_include_usage(&mut value) {
            replace_json_body(body, &value, "stream_usage_inject", "stream_options")
                .map_err(|e| -> FilterError { format!("stream_usage_inject: {e}").into() })?;
        }

        Ok(FilterAction::Continue)
    }
}
