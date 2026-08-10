// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! LLMISvc model-provider resolver: rewrites KServe publisher-ID body
//! `model` values to the short model name while leaving the routing
//! header untouched.

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_ai_apis::json_body::replace_json_body;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, PendingHeaderResult,
    body::DEFAULT_JSON_BODY_MAX_BYTES, builtins::http::payload_processing::config_validation::validate_max_body_bytes,
    parse_filter_config,
};
use serde::Deserialize;
use tracing::debug;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default header name for the routing model value (aligned with
/// [`super::ModelToHeaderFilter`]).
const DEFAULT_HEADER: &str = "X-Model";

/// Filter metadata key for the original publisher ID (for metering).
const META_PUBLISHER_ID: &str = "llmisvc_model_provider_resolver.publisher_id";

/// Prefix that identifies a KServe / LLMISvc publisher model ID.
const PUBLISHERS_PREFIX: &str = "publishers/";

/// Separator between the publisher path and the short model name.
const MODELS_SEPARATOR: &str = "/models/";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the LLMISvc model-provider resolver.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmisvcModelProviderResolverConfig {
    /// Request header that carries the publisher ID for KServe routing.
    ///
    /// Defaults to `X-Model` (same as `model_to_header`).
    #[serde(default = "default_header")]
    header: String,

    /// Maximum request body size to buffer before parsing.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// Default header name.
fn default_header() -> String {
    DEFAULT_HEADER.to_owned()
}

/// Default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

// -----------------------------------------------------------------------------
// LlmisvcModelProviderResolverFilter
// -----------------------------------------------------------------------------

/// Ports the LLMISvc / KServe BBR body-rewrite branch from IPP's
/// `model-provider-resolver`.
///
/// Prefer the configured request header (default `X-Model`) for the
/// model name, falling back to the JSON body `"model"` field. When the
/// resolved name is a publisher ID (`publishers/.../models/<name>`),
/// rewrite the body `"model"` to `<name>` only. The routing header is
/// never modified — KServe routes on the publisher ID.
///
/// Does **not** resolve ExternalModel / ExternalProvider CRDs, perform
/// weighted provider selection, rewrite `Host`, or inject credentials.
///
/// # YAML configuration
///
/// ```yaml
/// filter: llmisvc_model_provider_resolver
/// header: X-Model   # optional, defaults to X-Model
/// ```
///
/// # Example
///
/// ```ignore
/// use praxis_ai_filters::LlmisvcModelProviderResolverFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = LlmisvcModelProviderResolverFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "llmisvc_model_provider_resolver");
/// ```
pub struct LlmisvcModelProviderResolverFilter {
    /// Header that carries the publisher ID used for KServe routing.
    header: HeaderName,

    /// Maximum request body size to buffer.
    max_body_bytes: usize,
}

impl LlmisvcModelProviderResolverFilter {
    /// Create from parsed YAML config.
    ///
    /// Accepts an optional `header` field (defaults to `X-Model`) and
    /// optional `max_body_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing fails, `header` is
    /// empty/invalid, or `max_body_bytes` is invalid.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    ///
    /// ```ignore
    /// use praxis_ai_filters::LlmisvcModelProviderResolverFilter;
    ///
    /// let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
    /// let filter = LlmisvcModelProviderResolverFilter::from_config(&yaml).unwrap();
    /// assert_eq!(filter.name(), "llmisvc_model_provider_resolver");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: LlmisvcModelProviderResolverConfig = parse_filter_config("llmisvc_model_provider_resolver", config)?;

        let header = cfg.header.trim();
        if header.is_empty() {
            return Err("llmisvc_model_provider_resolver: 'header' must not be empty".into());
        }
        let header: HeaderName = header
            .parse()
            .map_err(|e| format!("llmisvc_model_provider_resolver: invalid 'header' name: {e}"))?;
        validate_max_body_bytes("llmisvc_model_provider_resolver", cfg.max_body_bytes)?;

        Ok(Box::new(Self {
            header,
            max_body_bytes: cfg.max_body_bytes,
        }))
    }

    /// Resolve model name, rewrite publisher-ID body field when needed.
    fn rewrite_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        let mut value: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(_) => return Ok(FilterAction::Continue),
        };

        let Some(obj) = value.as_object_mut() else {
            return Ok(FilterAction::Continue);
        };

        let Some(model_name) = resolve_model_name(ctx, &self.header, obj) else {
            return Ok(FilterAction::Continue);
        };

        let Some(short_name) = llmisvc_short_model_name(&model_name) else {
            return Ok(FilterAction::Continue);
        };

        // Stash the original publisher ID for later metering.
        ctx.set_metadata(META_PUBLISHER_ID, model_name.as_str());

        if obj.get("model").and_then(serde_json::Value::as_str) == Some(short_name) {
            return Ok(FilterAction::Continue);
        }

        obj.insert("model".to_owned(), serde_json::Value::String(short_name.to_owned()));

        replace_json_body(body, &value, self.name(), "model").map_err(|e| -> FilterError {
            format!("{}: failed to re-serialize rewritten request body: {e}", self.name()).into()
        })?;

        debug!(
            original = %model_name,
            rewritten = %short_name,
            "LLMISvc BBR: rewrote body model field"
        );

        Ok(FilterAction::Continue)
    }
}

#[async_trait]
impl HttpFilter for LlmisvcModelProviderResolverFilter {
    fn name(&self) -> &'static str {
        "llmisvc_model_provider_resolver"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
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

        self.rewrite_body(ctx, body)
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Prefer the configured header, then the body `"model"` string.
fn resolve_model_name(
    ctx: &HttpFilterContext<'_>,
    header: &HeaderName,
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    if let Some(from_header) = header_model_name(ctx, header) {
        return Some(from_header);
    }

    obj.get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

/// Read a non-empty model name from the request headers or pending
/// mutations (e.g. `extra_request_headers` from an earlier
/// `model_to_header`).
fn header_model_name(ctx: &HttpFilterContext<'_>, header: &HeaderName) -> Option<String> {
    if let Some(value) = ctx.request.headers.get(header)
        && let Ok(s) = value.to_str()
    {
        let s = s.trim();
        if !s.is_empty() {
            return Some(s.to_owned());
        }
    }

    match ctx.pending_header_value(header) {
        Ok(PendingHeaderResult::Value(value)) => {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_owned())
        },
        Ok(PendingHeaderResult::Absent | PendingHeaderResult::Removed) | Err(_) => None,
    }
}

/// Extract the short model name from a KServe publisher ID.
///
/// Mirrors IPP: require `publishers/` prefix, then take the segment
/// after the first `/models/` when non-empty.
fn llmisvc_short_model_name(model_name: &str) -> Option<&str> {
    if !model_name.starts_with(PUBLISHERS_PREFIX) {
        return None;
    }
    let (_, short_name) = model_name.split_once(MODELS_SEPARATOR)?;
    if short_name.is_empty() { None } else { Some(short_name) }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::borrow::Cow;

    use http::HeaderValue;

    use super::*;

    fn filter_default() -> Box<dyn HttpFilter> {
        LlmisvcModelProviderResolverFilter::from_config(&serde_yaml::Value::Null).unwrap()
    }

    #[test]
    fn from_config_default_header() {
        let filter = filter_default();
        assert_eq!(
            filter.name(),
            "llmisvc_model_provider_resolver",
            "default config should produce llmisvc_model_provider_resolver"
        );
    }

    #[test]
    fn from_config_custom_header() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
        let filter = LlmisvcModelProviderResolverFilter::from_config(&yaml).unwrap();
        assert_eq!(
            filter.name(),
            "llmisvc_model_provider_resolver",
            "custom header config should parse"
        );
    }

    #[test]
    fn from_config_rejects_empty_header() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: \"\"").unwrap();
        match LlmisvcModelProviderResolverFilter::from_config(&yaml) {
            Err(err) => assert!(
                err.to_string().contains("header"),
                "empty header should be rejected: {err}"
            ),
            Ok(_) => panic!("empty header should be rejected"),
        }
    }

    #[test]
    fn from_config_rejects_zero_max_body_bytes() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
        match LlmisvcModelProviderResolverFilter::from_config(&yaml) {
            Err(err) => assert!(
                err.to_string().contains("max_body_bytes"),
                "zero max_body_bytes should be rejected: {err}"
            ),
            Ok(_) => panic!("zero max_body_bytes should be rejected"),
        }
    }

    #[test]
    fn body_access_is_read_write_stream_buffer() {
        let filter = filter_default();
        assert_eq!(
            filter.request_body_access(),
            BodyAccess::ReadWrite,
            "must mutate the request body"
        );
        assert!(
            matches!(
                filter.request_body_mode(),
                BodyMode::StreamBuffer {
                    max_bytes: Some(limit)
                } if limit > 0
            ),
            "body mode should be StreamBuffer with a default size limit"
        );
    }

    #[test]
    fn short_model_name_extracts_after_models() {
        assert_eq!(
            llmisvc_short_model_name("publishers/ns/models/granite-3.1-8b"),
            Some("granite-3.1-8b")
        );
        assert_eq!(
            llmisvc_short_model_name("publishers/ns/models/a/b"),
            Some("a/b"),
            "SplitN keeps remainder after first /models/"
        );
        assert_eq!(llmisvc_short_model_name("publishers/ns/models/"), None);
        assert_eq!(llmisvc_short_model_name("publishers/ns/foo"), None);
        assert_eq!(llmisvc_short_model_name("granite-3.1-8b"), None);
        assert_eq!(llmisvc_short_model_name("other/models/foo"), None);
    }

    #[tokio::test]
    async fn rewrites_body_model_from_header_publisher_id() {
        let filter = filter_default();
        let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        req.headers.insert(
            "X-Model",
            HeaderValue::from_static("publishers/rhoai/models/granite-3.1-8b"),
        );
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/rhoai/models/granite-3.1-8b","messages":[]}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "rewrite should continue");

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(
            parsed["model"].as_str(),
            Some("granite-3.1-8b"),
            "body model should be rewritten to short name"
        );
        assert_eq!(
            ctx.filter_metadata.get(META_PUBLISHER_ID).map(String::as_str),
            Some("publishers/rhoai/models/granite-3.1-8b"),
            "publisher ID should be stashed in filter metadata"
        );
        assert!(
            ctx.extra_request_headers.is_empty(),
            "routing header must not be modified"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "routing header must not be overwritten via request_headers_to_set"
        );
        assert!(
            ctx.request_headers_to_remove.is_empty(),
            "routing header must not be removed"
        );
    }

    #[tokio::test]
    async fn falls_back_to_body_model_when_header_absent() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/ns/models/mistral","prompt":"hi"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(parsed["model"].as_str(), Some("mistral"));
        assert_eq!(parsed["prompt"].as_str(), Some("hi"), "other fields preserved");
    }

    #[tokio::test]
    async fn prefers_header_over_body_model() {
        let filter = filter_default();
        let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        req.headers
            .insert("X-Model", HeaderValue::from_static("publishers/ns/models/from-header"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/ns/models/from-body"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(parsed["model"].as_str(), Some("from-header"));
        assert_eq!(
            ctx.filter_metadata.get(META_PUBLISHER_ID).map(String::as_str),
            Some("publishers/ns/models/from-header"),
        );
    }

    #[tokio::test]
    async fn reads_pending_extra_request_headers() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.extra_request_headers
            .push((Cow::Borrowed("X-Model"), "publishers/ns/models/via-extra".to_owned()));

        let json = br#"{"model":"publishers/ns/models/via-extra"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(parsed["model"].as_str(), Some("via-extra"));
        assert_eq!(
            ctx.extra_request_headers.len(),
            1,
            "extra header from model_to_header must remain"
        );
        assert_eq!(ctx.extra_request_headers[0].1, "publishers/ns/models/via-extra");
    }

    #[tokio::test]
    async fn leaves_non_publisher_model_unchanged() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"mistral-large-latest"}"#;
        let mut body = Some(Bytes::from_static(json));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "non-publisher body must not be rewritten");
        assert!(
            !ctx.filter_metadata.contains_key(META_PUBLISHER_ID),
            "no publisher metadata for non-publisher models"
        );
    }

    #[tokio::test]
    async fn continues_when_model_absent() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"messages":[]}"#;
        let mut body = Some(Bytes::from_static(json));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original);
    }

    #[tokio::test]
    async fn continues_on_invalid_json() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let mut body = Some(Bytes::from_static(b"not-json"));
        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body.as_deref(), Some(b"not-json".as_slice()));
    }

    #[tokio::test]
    async fn custom_header_name_used() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
        let filter = LlmisvcModelProviderResolverFilter::from_config(&yaml).unwrap();
        let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        req.headers
            .insert("X-AI-Model", HeaderValue::from_static("publishers/ns/models/custom"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/ns/models/custom"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(parsed["model"].as_str(), Some("custom"));
    }

    #[tokio::test]
    async fn on_request_is_noop() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }

    #[tokio::test]
    async fn waits_for_end_of_stream() {
        let filter = filter_default();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"publishers/ns/models/granite"}"#;
        let mut body = Some(Bytes::from_static(json));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "must not rewrite before end_of_stream");
    }
}
