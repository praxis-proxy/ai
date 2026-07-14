// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Model-to-header filter: promotes the "model" JSON body field to a request header for routing.

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, builtins::JsonBodyFieldFilter,
    parse_filter_config,
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
    #[serde(default = "default_header")]
    header: String,
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
/// # YAML configuration
///
/// ```yaml
/// filter: model_to_header
/// header: X-Model   # optional, defaults to X-Model
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
}

impl ModelToHeaderFilter {
    /// Create from parsed YAML config.
    ///
    /// Accepts an optional `header` field (defaults to `X-Model`).
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the inner `JsonBodyFieldFilter` config is invalid.
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
        let header = &cfg.header;

        let mut inner_config = serde_yaml::Mapping::new();
        inner_config.insert(
            serde_yaml::Value::String("field".into()),
            serde_yaml::Value::String("model".into()),
        );
        inner_config.insert(
            serde_yaml::Value::String("header".into()),
            serde_yaml::Value::String(header.to_owned()),
        );

        let inner = JsonBodyFieldFilter::from_config(&serde_yaml::Value::Mapping(inner_config))?;

        Ok(Box::new(Self { inner }))
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
    use super::*;

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
    async fn extracts_model_field() {
        let filter = ModelToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"mistral-large-latest","prompt":"hello"}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Release),
            "should release after extracting model"
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
    async fn custom_header_name_used() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("header: X-AI-Model").unwrap();
        let filter = ModelToHeaderFilter::from_config(&yaml).unwrap();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let json = br#"{"model":"claude-3","messages":[]}"#;
        let mut body = Some(Bytes::from_static(json));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

        assert!(
            matches!(action, FilterAction::Release),
            "should release after extracting model"
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
}
