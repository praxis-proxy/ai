// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Model-to-header filter: promotes the "model" JSON body field to a request header for routing.

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, PendingHeaderResult,
    builtins::JsonBodyFieldFilter, parse_filter_config,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default header name for the promoted model value.
const DEFAULT_HEADER: &str = "X-Model";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the model-to-header filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelToHeaderConfig {
    /// Header name for the promoted model value.
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// internal `x-praxis-*` header. Defaults to `X-Model`.
    #[serde(default = "default_header")]
    header: String,

    /// Keep a model header the request already carries and skip the body
    /// parse. Defaults to `false`.
    ///
    /// When `true` and the request already has a single non-empty `header`
    /// (from the client or from an earlier filter), the filter leaves it as is:
    /// it is not stripped, the body is not parsed, and nothing is promoted. The
    /// value is trusted exactly as it arrived and is never compared with the
    /// body's `model`, so enable this only when a trusted hop in front of
    /// Praxis sets the header. The body is still buffered.
    #[serde(default)]
    trust_existing_header: bool,
}

/// Default header name.
fn default_header() -> String {
    DEFAULT_HEADER.to_owned()
}

// -----------------------------------------------------------------------------
// ModelToHeaderFilter
// -----------------------------------------------------------------------------

/// Promotes the JSON `"model"` field from the request body to a request header.
///
/// Promotion is deferred until end-of-stream so a later body-writing filter
/// (for example `llmisvc_model_provider_resolver`) can observe the pending
/// header in the same `StreamBuffer` pre-read pass.
///
/// With `trust_existing_header: true`, a request that already carries the
/// header keeps it and skips the body parse. That turns off the anti-spoofing
/// strip for this filter, so whoever sends the header picks the model: only
/// enable it when a trusted hop in front of Praxis sets or scrubs the header.
/// The body is still buffered, because the pipeline fixes its body mode when it
/// is built, not per request.
///
/// # YAML configuration
///
/// ```yaml
/// filter: model_to_header
/// header: X-Model               # optional, defaults to X-Model
/// trust_existing_header: false  # optional; true keeps an existing header
/// ```
///
/// # Example
///
/// ```ignore
/// use praxis_ai_filters::ModelToHeaderFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = ModelToHeaderFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "model_to_header");
/// ```
pub struct ModelToHeaderFilter {
    /// Delegated body-field extraction filter (type-erased
    /// `JsonBodyFieldFilter`).
    inner: Box<dyn HttpFilter>,
    /// The promotion-target header this filter owns; a client-supplied copy is
    /// stripped before promotion so routing cannot be spoofed. See #1039.
    header: HeaderName,
    /// Operator opt-in to keep an existing `header` and skip the body parse,
    /// which turns off the strip above for requests that carry one.
    trust_existing_header: bool,
}

impl ModelToHeaderFilter {
    /// Create from parsed YAML config.
    ///
    /// Accepts an optional `header` field (defaults to `X-Model`) and an
    /// optional `trust_existing_header` flag (defaults to `false`).
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the header name is unsafe or the inner
    /// `JsonBodyFieldFilter` config is invalid.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    ///
    /// ```ignore
    /// use praxis_ai_filters::ModelToHeaderFilter;
    ///
    /// let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
    /// let filter = ModelToHeaderFilter::from_config(&yaml).unwrap();
    /// assert_eq!(filter.name(), "model_to_header");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ModelToHeaderConfig = parse_filter_config("model_to_header", config)?;
        let header =
            praxis_ai_apis::promotion::parse_dedicated_promotion_header("model_to_header", "header", &cfg.header, &[])?;

        let mut inner_config = serde_yaml::Mapping::new();
        inner_config.insert(
            serde_yaml::Value::String("field".into()),
            serde_yaml::Value::String("model".into()),
        );
        inner_config.insert(
            serde_yaml::Value::String("header".into()),
            serde_yaml::Value::String(cfg.header.clone()),
        );

        let inner = JsonBodyFieldFilter::from_config(&serde_yaml::Value::Mapping(inner_config))?;

        Ok(Box::new(Self {
            inner,
            header,
            trust_existing_header: cfg.trust_existing_header,
        }))
    }
}

#[async_trait]
impl HttpFilter for ModelToHeaderFilter {
    fn name(&self) -> &'static str {
        "model_to_header"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        self.inner.on_request(ctx).await
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        self.inner.on_response(ctx).await
    }

    fn request_body_access(&self) -> BodyAccess {
        self.inner.request_body_access()
    }

    fn response_body_access(&self) -> BodyAccess {
        self.inner.response_body_access()
    }

    fn request_body_mode(&self) -> BodyMode {
        self.inner.request_body_mode()
    }

    fn response_body_mode(&self) -> BodyMode {
        self.inner.response_body_mode()
    }

    fn needs_request_context(&self) -> bool {
        self.inner.needs_request_context()
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
        if self.trust_existing_header && header_already_set(ctx, &self.header) {
            tracing::debug!(
                header = %self.header,
                "model_to_header: keeping existing header and skipping the body parse (trust_existing_header)"
            );
            return Ok(FilterAction::Continue);
        }
        if !ctx.request_headers_to_remove.contains(&self.header) {
            ctx.request_headers_to_remove.push(self.header.clone());
            tracing::debug!(
                header = %self.header,
                "model_to_header: dropping client-supplied promotion header (anti-spoofing)"
            );
        }
        self.inner.on_request_body(ctx, body, end_of_stream).await
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        self.inner.on_response_body(ctx, body, end_of_stream)
    }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Whether the request already carries a usable `header`, as the upstream
/// would see it once pending pre-read mutations are applied.
///
/// Resolves in the same last-writer-wins order as Praxis's effective-header
/// view for body-phase conditions: this pass's grouped queues, then the
/// ordered pre-read log, then the original request. So an earlier filter's
/// removal hides a client copy, and an earlier filter's value counts even when
/// the client sent none. Anything ambiguous counts as absent, which falls back
/// to the stripping path.
fn header_already_set(ctx: &HttpFilterContext<'_>, header: &HeaderName) -> bool {
    pending_header_state(ctx, header)
        .or_else(|| trusted_log_state(ctx, header))
        .unwrap_or_else(|| original_header_set(ctx, header))
}

/// Presence per this pass's grouped mutation queues, or `None` when they do
/// not mention `header`.
fn pending_header_state(ctx: &HttpFilterContext<'_>, header: &HeaderName) -> Option<bool> {
    match ctx.pending_header_value(header) {
        Ok(PendingHeaderResult::Value(value)) => Some(is_usable_value(&value)),
        Ok(PendingHeaderResult::Removed) | Err(_) => Some(false),
        Ok(PendingHeaderResult::Absent) => None,
    }
}

/// Presence per the ordered pre-read log (earlier passes, then this one), or
/// `None` when no entry touches `header`.
fn trusted_log_state(ctx: &HttpFilterContext<'_>, header: &HeaderName) -> Option<bool> {
    let touched = ctx
        .prior_pre_read_mutations
        .iter()
        .chain(&ctx.pre_read_mutations)
        .any(|mutation| mutation.matches_header(header));

    touched.then(|| matches!(ctx.resolve_trusted_header(header), Ok(Some(value)) if is_usable_value(&value)))
}

/// Whether the original request carries exactly one non-blank text `header`.
fn original_header_set(ctx: &HttpFilterContext<'_>, header: &HeaderName) -> bool {
    let mut values = ctx.request.headers.get_all(header).iter();
    match (values.next(), values.next()) {
        (Some(value), None) => value.to_str().is_ok_and(is_usable_value),
        _ => false,
    }
}

/// Whether a header value has anything in it besides whitespace.
fn is_usable_value(value: &str) -> bool {
    !value.trim().is_empty()
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
    use http::HeaderValue;
    use praxis_filter::TrustedHeaderMutation;

    use super::*;

    /// Request body naming a different model than the headers the tests send.
    const LLAMA_BODY: &[u8] = br#"{"model":"llama-3.2-8b","messages":[]}"#;

    #[test]
    fn from_config_default_header() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(
            filter.name(),
            "model_to_header",
            "default config should produce model_to_header"
        );
    }

    #[test]
    fn from_config_custom_header() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
        let filter = ModelToHeaderFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.name(), "model_to_header", "custom header config should parse");
    }

    #[test]
    fn from_config_rejects_api_key_header() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: x-api-key").unwrap();
        let err = ModelToHeaderFilter::from_config(&yaml)
            .err()
            .expect("x-api-key should be rejected");
        assert!(
            err.to_string().contains("x-api-key"),
            "x-api-key promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn from_config_rejects_format_routing_header() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: x-praxis-ai-format").unwrap();
        let err = ModelToHeaderFilter::from_config(&yaml)
            .err()
            .expect("x-praxis-ai-format should be rejected");
        assert!(
            err.to_string().contains("x-praxis-ai-format"),
            "x-praxis-ai-format promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn body_access_delegates_to_inner() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(
            filter.request_body_access(),
            BodyAccess::ReadOnly,
            "body access should delegate to inner"
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

    #[tokio::test]
    async fn waits_for_end_of_stream() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"mistral-large-latest","prompt":"hello"}"#;
        let mut body = Some(Bytes::from_static(json));
        let original = body.clone();

        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, original, "must not promote before end_of_stream");
        assert!(ctx.extra_request_headers.is_empty());
    }

    #[tokio::test]
    async fn extracts_model_field() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"mistral-large-latest","prompt":"hello"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::BodyDone),
            "should complete with BodyDone after extracting model"
        );
        assert_eq!(ctx.extra_request_headers.len(), 1, "should add exactly one header");
        let (name, value) = &ctx.extra_request_headers[0];
        assert_eq!(name, "X-Model", "header name should be X-Model");
        assert_eq!(
            value, "mistral-large-latest",
            "model value should be promoted to X-Model header"
        );
    }

    #[tokio::test]
    async fn strips_client_header_alongside_promotion() {
        // A body-derived promotion must queue a Remove of the owned header in
        // the same pre-read pass as the Add, so the pass's remove -> set -> add
        // application drops any spoofed client copy. See praxis-proxy/ai#1039.
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"llama-3.2-8b","messages":[]}"#;
        let mut body = Some(Bytes::from_static(json));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            ctx.request_headers_to_remove.iter().any(|h| h == "x-model"),
            "the owned header must be queued for removal before promotion"
        );
        let (name, value) = &ctx.extra_request_headers[0];
        assert_eq!(name, "X-Model", "the body-derived value must still be promoted");
        assert_eq!(value, "llama-3.2-8b", "the promoted value must be the body model");
    }

    #[tokio::test]
    async fn strips_client_header_even_without_body_model() {
        // Fail-closed: the client header is never trusted, so it is removed
        // even when the body carries no model to promote.
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"prompt":"hello"}"#;
        let mut body = Some(Bytes::from_static(json));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            ctx.request_headers_to_remove.iter().any(|h| h == "x-model"),
            "the owned header must be removed even when no body model is promoted"
        );
        assert!(
            ctx.extra_request_headers.is_empty(),
            "no header is promoted when the body has no model"
        );
    }

    #[tokio::test]
    async fn custom_header_name_used() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
        let filter = ModelToHeaderFilter::from_config(&yaml).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"claude-3","messages":[]}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::BodyDone),
            "should complete with BodyDone after extracting model"
        );
        let (name, value) = &ctx.extra_request_headers[0];
        assert_eq!(name, "X-AI-Model", "header name should be X-AI-Model");
        assert_eq!(value, "claude-3", "model should be promoted to custom header name");
    }

    #[tokio::test]
    async fn continues_when_model_absent() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"prompt":"hello"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "absent model field should continue"
        );
        assert!(
            ctx.extra_request_headers.is_empty(),
            "no headers when model field absent"
        );
    }

    #[tokio::test]
    async fn on_request_is_noop() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue), "on_request should be a no-op");
    }

    #[tokio::test]
    async fn on_response_delegates_to_inner() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut resp = crate::test_utils::make_response();
        ctx.response_header = Some(&mut resp);

        let action = filter.on_response(&mut ctx).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "on_response should delegate to inner and return Continue"
        );
    }

    #[test]
    fn response_body_access_delegates_to_inner() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(
            filter.response_body_access(),
            BodyAccess::None,
            "response body access should delegate to inner"
        );
    }

    #[test]
    fn response_body_mode_delegates_to_inner() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert!(
            matches!(filter.response_body_mode(), BodyMode::Stream),
            "response body mode should delegate to inner"
        );
    }

    #[test]
    fn needs_request_context_delegates_to_inner() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        // JsonBodyFieldFilter does not need request context on the
        // response path; verify the delegate forwards the inner value.
        assert!(
            !filter.needs_request_context(),
            "needs_request_context should delegate to inner and return false"
        );
    }

    #[test]
    fn on_response_body_delegates_to_inner() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut body = Some(Bytes::from_static(b"response data"));

        let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "on_response_body should delegate to inner and return Continue"
        );
    }

    #[test]
    fn config_trust_existing_header_defaults_off() {
        let cfg: ModelToHeaderConfig = serde_yaml::from_str("header: X-Model").unwrap();
        assert!(
            !cfg.trust_existing_header,
            "trust_existing_header must default to false so the strip stays on"
        );
    }

    #[test]
    fn config_trust_existing_header_parses() {
        let cfg: ModelToHeaderConfig = serde_yaml::from_str("trust_existing_header: true").unwrap();
        assert!(cfg.trust_existing_header, "trust_existing_header: true should parse");
        assert_eq!(cfg.header, DEFAULT_HEADER, "header should keep its default");
    }

    #[test]
    fn config_rejects_misspelled_trust_field() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("trust_existing_headers: true").unwrap();
        let err = ModelToHeaderFilter::from_config(&yaml)
            .err()
            .expect("a misspelled field should be rejected");
        assert!(
            err.to_string().contains("trust_existing_headers"),
            "deny_unknown_fields should name the unknown field: {err}"
        );
    }

    #[tokio::test]
    async fn trust_existing_header_keeps_client_header_and_skips_parse() {
        let filter = trusting_filter();
        let req = request_with_model_headers(&[b"granite-3.3-8b"]);
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "a trusted existing header should continue"
        );
        assert!(
            ctx.request_headers_to_remove.is_empty(),
            "the existing header must not be queued for removal"
        );
        assert!(
            ctx.extra_request_headers.is_empty(),
            "the body model must not be promoted over the existing header"
        );
        assert_eq!(
            body.as_deref(),
            Some(LLAMA_BODY),
            "the body must pass through untouched"
        );
    }

    #[tokio::test]
    async fn trust_existing_header_keeps_pending_value_from_earlier_filter() {
        let filter = trusting_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.request_headers_to_set
            .push((model_header(), HeaderValue::from_static("granite-3.3-8b")));
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            ctx.request_headers_to_remove.is_empty(),
            "a value an earlier filter set this pass must not be stripped"
        );
        assert!(
            ctx.extra_request_headers.is_empty(),
            "a value an earlier filter set this pass must not be overridden"
        );
    }

    #[tokio::test]
    async fn trust_existing_header_keeps_value_from_earlier_pass() {
        let filter = trusting_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.prior_pre_read_mutations
            .push(TrustedHeaderMutation::Add(model_header(), "granite-3.3-8b".to_owned()));
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            ctx.request_headers_to_remove.is_empty(),
            "a value an earlier pre-read pass promoted must not be stripped"
        );
        assert!(
            ctx.extra_request_headers.is_empty(),
            "a value an earlier pre-read pass promoted must not be overridden"
        );
    }

    #[tokio::test]
    async fn trust_existing_header_promotes_after_pending_removal() {
        let filter = trusting_filter();
        let req = request_with_model_headers(&[b"granite-3.3-8b"]);
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.request_headers_to_remove.push(model_header());
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert_promoted_from_body(&ctx, "a header an earlier filter removed this pass is absent");
    }

    #[tokio::test]
    async fn trust_existing_header_promotes_after_logged_removal() {
        let filter = trusting_filter();
        let req = request_with_model_headers(&[b"granite-3.3-8b"]);
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.prior_pre_read_mutations
            .push(TrustedHeaderMutation::Remove(model_header()));
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert_promoted_from_body(&ctx, "a header an earlier pre-read pass removed is absent");
    }

    #[tokio::test]
    async fn trust_existing_header_promotes_when_header_missing() {
        let filter = trusting_filter();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::BodyDone),
            "promotion should complete with BodyDone"
        );
        assert_promoted_from_body(&ctx, "with no header the body model is promoted");
    }

    #[tokio::test]
    async fn trust_existing_header_ignores_unusable_header_values() {
        let cases: [(&[&[u8]], &str); 4] = [
            (&[b""], "empty"),
            (&[b"   "], "whitespace-only"),
            (&[b"\xff\xfe"], "non-UTF-8"),
            (&[b"granite-3.3-8b", b"llama-3.2-8b"], "multi-valued"),
        ];
        for (values, label) in cases {
            let filter = trusting_filter();
            let req = request_with_model_headers(values);
            let mut ctx = crate::test_utils::make_filter_context(&req);
            let mut body = Some(Bytes::from_static(LLAMA_BODY));

            let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

            assert_promoted_from_body(&ctx, &format!("a {label} header counts as absent"));
        }
    }

    #[tokio::test]
    async fn trust_existing_header_off_strips_client_header() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("trust_existing_header: false").unwrap();
        let filter = ModelToHeaderFilter::from_config(&yaml).unwrap();
        let req = request_with_model_headers(&[b"granite-3.3-8b"]);
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let mut body = Some(Bytes::from_static(LLAMA_BODY));

        let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert_promoted_from_body(&ctx, "with the knob off a client header is never trusted");
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// A filter with `trust_existing_header` on and the default header.
    fn trusting_filter() -> Box<dyn HttpFilter> {
        let yaml: serde_yaml::Value = serde_yaml::from_str("trust_existing_header: true").unwrap();
        ModelToHeaderFilter::from_config(&yaml).unwrap()
    }

    /// The default promotion header, parsed.
    fn model_header() -> HeaderName {
        HeaderName::from_static("x-model")
    }

    /// A POST carrying one `X-Model` line per entry in `values`.
    fn request_with_model_headers(values: &[&[u8]]) -> praxis_filter::Request {
        let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        for value in values {
            req.headers
                .append(model_header(), HeaderValue::from_bytes(value).unwrap());
        }
        req
    }

    /// Assert the filter took the stripping path: the header is queued for
    /// removal and the body's model is promoted in its place.
    fn assert_promoted_from_body(ctx: &HttpFilterContext<'_>, why: &str) {
        assert!(
            ctx.request_headers_to_remove.contains(&model_header()),
            "{why}: the header should be queued for removal"
        );
        assert_eq!(
            ctx.extra_request_headers.len(),
            1,
            "{why}: the body model should be promoted"
        );
        let (name, value) = &ctx.extra_request_headers[0];
        assert!(name.eq_ignore_ascii_case("x-model"), "{why}: promoted to X-Model");
        assert_eq!(value, "llama-3.2-8b", "{why}: the promoted value is the body model");
    }
}
