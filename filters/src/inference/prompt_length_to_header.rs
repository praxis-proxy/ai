// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Prompt-length routing fact derived from provider-neutral request content.

use std::{borrow::Cow, collections::HashMap, fmt};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_ai_apis::promotion::{is_promotable_value, parse_dedicated_promotion_header};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::{
    Deserialize, Deserializer,
    de::{IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;

/// Default header carrying the configured overflow route key.
const DEFAULT_HEADER: &str = "X-Prompt-Route";
/// Conservative default for ordinary ASCII prompt text.
const ASCII_CHARS_PER_TOKEN: usize = 3;
/// Maximum number of model-specific routing rules.
const MAX_ROUTES: usize = 1024;
/// Metadata key carrying the bounded numeric estimate.
const META_ESTIMATED_TOKENS: &str = "prompt_length_to_header.estimated_tokens";
/// Metadata key carrying the selected overflow route.
const META_OVERFLOW_ROUTE: &str = "prompt_length_to_header.overflow_route";
/// Prompt-bearing fields whose nested string values contribute to the estimate.
const PROMPT_FIELDS: [&str; 5] = ["messages", "prompt", "input", "system", "instructions"];
/// Schema-bearing fields whose nested keys and string values contribute.
const SCHEMA_FIELDS: [&str; 6] = [
    "tools",
    "functions",
    "response_format",
    "text",
    "output_config",
    "output_format",
];

/// Deserialized configuration for prompt-length routing.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptLengthToHeaderConfig {
    /// Per-model input limits and overflow route keys.
    routes: Vec<PromptLengthRouteConfig>,
    /// Header that receives an overflow route key. Defaults to `X-Prompt-Route`.
    #[serde(default = "default_header")]
    header: String,
    /// Maximum JSON request body buffered for estimation.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// One model-specific prompt limit and overflow route key.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptLengthRouteConfig {
    /// Exact request-body model name.
    model: String,
    /// Largest estimated input-token count that remains on the default route.
    max_input_tokens: usize,
    /// Bounded value promoted when the estimate exceeds `max_input_tokens`.
    overflow: String,
}

/// Validated route rule indexed by request model.
#[derive(Debug)]
struct PromptLengthRoute {
    /// Maximum estimated input tokens accepted by the default route.
    max_input_tokens: usize,
    /// Header value consumed by an exact-match routing rule.
    overflow: String,
}

/// Borrowed model lookup used to bypass full JSON allocation for unknown models.
struct ModelProbe<'a>(Option<Cow<'a, str>>);

/// A string value when present, or `None` after consuming any other JSON value.
struct OptionalString<'a>(Option<Cow<'a, str>>);

/// Estimates prompt length and promotes a route key when a model limit is exceeded.
///
/// The estimate is deliberately provider-neutral and conservative. It counts
/// text in common prompt-bearing and schema-bearing fields: `messages`,
/// `prompt`, `input`, `system`, `instructions`, `tools`, `functions`,
/// `response_format`, `text`, `output_config`, and `output_format`. ASCII text
/// is rounded up at three characters per token; non-ASCII text is charged at
/// one token per UTF-8 byte so emoji and CJK input cannot be silently
/// underestimated. Object keys under schema-bearing fields are also counted so
/// JSON Schema property names contribute to the estimate. It is a routing
/// heuristic, not a billing or quota token count.
///
/// Place body-writing filters such as `prompt_enrich` before this filter. The
/// estimate reflects the body produced by earlier filters in the chain, so the
/// recommended order is body writers, `prompt_length_to_header`, then `router`.
///
/// The filter removes any client-supplied copy of its output header before it
/// publishes a derived value. A following core `router` can exact-match the
/// configured `overflow` key and select a larger-context cluster.
///
/// # YAML configuration
///
/// ```yaml
/// filter: prompt_length_to_header
/// header: X-Prompt-Route
/// routes:
///   - model: small-context-model
///     max_input_tokens: 8192
///     overflow: large-context
/// ```
pub struct PromptLengthToHeaderFilter {
    /// Model-specific prompt limits.
    routes: HashMap<String, PromptLengthRoute>,
    /// Owned promotion header, removed before any derived value is appended.
    header: HeaderName,
    /// Maximum buffered request size.
    max_body_bytes: usize,
}

impl PromptLengthToHeaderFilter {
    /// Construct the filter from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or duplicate route table, unsafe routing
    /// values, zero token limits, an unsafe header, or an invalid body limit.
    ///
    /// # Example
    ///
    /// ```
    /// use praxis_ai_filters::PromptLengthToHeaderFilter;
    ///
    /// let config = serde_yaml::from_str(
    ///     "routes:\n  - model: small-model\n    max_input_tokens: 8192\n    overflow: large-context",
    /// )
    /// .unwrap();
    /// let filter = PromptLengthToHeaderFilter::from_config(&config).unwrap();
    /// assert_eq!(filter.name(), "prompt_length_to_header");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: PromptLengthToHeaderConfig = parse_filter_config("prompt_length_to_header", config)?;
        validate_max_body_bytes("prompt_length_to_header", config.max_body_bytes)?;
        let header = parse_dedicated_promotion_header("prompt_length_to_header", "header", &config.header, &[])?;
        Ok(Box::new(Self {
            routes: build_routes(config.routes)?,
            header,
            max_body_bytes: config.max_body_bytes,
        }))
    }
}

#[async_trait]
impl HttpFilter for PromptLengthToHeaderFilter {
    fn name(&self) -> &'static str {
        "prompt_length_to_header"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
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

        if !ctx.request_headers_to_remove.contains(&self.header) {
            ctx.request_headers_to_remove.push(self.header.clone());
        }
        ctx.filter_metadata.remove(META_ESTIMATED_TOKENS);
        ctx.filter_metadata.remove(META_OVERFLOW_ROUTE);

        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(ModelProbe(Some(model))) = serde_json::from_slice::<ModelProbe<'_>>(raw) else {
            return Ok(FilterAction::Continue);
        };
        let Some(route) = self.routes.get(model.as_ref()) else {
            return Ok(FilterAction::Continue);
        };
        let Ok(value) = serde_json::from_slice::<Value>(raw) else {
            return Ok(FilterAction::Continue);
        };

        let estimated_tokens = estimate_prompt_tokens(&value);
        ctx.set_metadata(META_ESTIMATED_TOKENS, estimated_tokens.to_string());
        if estimated_tokens > route.max_input_tokens {
            ctx.extra_request_headers
                .push((Cow::Owned(self.header.as_str().to_owned()), route.overflow.clone()));
            ctx.set_metadata(META_OVERFLOW_ROUTE, route.overflow.clone());
        }

        Ok(FilterAction::Continue)
    }
}

impl<'de> Deserialize<'de> for ModelProbe<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ModelProbeVisitor)
    }
}

/// Visits only the top-level model field and skips every unrelated value.
struct ModelProbeVisitor;

impl<'de> Visitor<'de> for ModelProbeVisitor {
    type Value = ModelProbe<'de>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object containing an optional model field")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut model = None;
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            if key == "model" {
                model = map.next_value::<OptionalString<'de>>()?.0;
            } else {
                let _ = map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(ModelProbe(model))
    }
}

impl<'de> Deserialize<'de> for OptionalString<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OptionalStringVisitor)
    }
}

/// Retains a borrowed string and consumes non-string JSON without allocation.
struct OptionalStringVisitor;

impl<'de> Visitor<'de> for OptionalStringVisitor {
    type Value = OptionalString<'de>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(OptionalString(Some(Cow::Borrowed(v))))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
        Ok(OptionalString(Some(Cow::Owned(v.to_owned()))))
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
        Ok(OptionalString(Some(Cow::Owned(v))))
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(OptionalString(None))
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(OptionalString(None))
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(OptionalString(None))
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(OptionalString(None))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(OptionalString(None))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(OptionalString(None))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(OptionalString(None))
    }
}

/// Default promotion header.
fn default_header() -> String {
    DEFAULT_HEADER.to_owned()
}

/// Default request body limit shared by JSON body filters.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Validate and index the model-specific overflow routes.
fn build_routes(configs: Vec<PromptLengthRouteConfig>) -> Result<HashMap<String, PromptLengthRoute>, FilterError> {
    if configs.is_empty() || configs.len() > MAX_ROUTES {
        return Err(format!("prompt_length_to_header: routes must contain 1-{MAX_ROUTES} entries").into());
    }
    let mut routes = HashMap::with_capacity(configs.len());
    for route in configs {
        validate_route_value("model", &route.model)?;
        validate_route_value("overflow", &route.overflow)?;
        if route.max_input_tokens == 0 {
            return Err("prompt_length_to_header: max_input_tokens must be greater than zero".into());
        }
        let model = route.model.clone();
        let replaced = routes.insert(
            route.model,
            PromptLengthRoute {
                max_input_tokens: route.max_input_tokens,
                overflow: route.overflow,
            },
        );
        if replaced.is_some() {
            return Err(format!("prompt_length_to_header: duplicate model '{model}'").into());
        }
    }
    Ok(routes)
}

/// Validate a configured routing value before it can become metadata or a header.
fn validate_route_value(field: &str, value: &str) -> Result<(), FilterError> {
    if value.is_empty() || !is_promotable_value(value) {
        return Err(
            format!("prompt_length_to_header: {field} must be non-empty, control-free, and at most 256 bytes").into(),
        );
    }
    Ok(())
}

/// Estimate input tokens from provider-neutral prompt-bearing fields.
fn estimate_prompt_tokens(body: &Value) -> usize {
    let prompt_tokens = PROMPT_FIELDS
        .into_iter()
        .filter_map(|field| body.get(field))
        .fold(0_usize, |total, value| {
            total.saturating_add(string_token_estimate(value, false))
        });
    SCHEMA_FIELDS
        .into_iter()
        .filter_map(|field| body.get(field))
        .fold(prompt_tokens, |total, value| {
            total.saturating_add(string_token_estimate(value, true))
        })
}

/// Estimate tokens in every string and, for schemas, every nested object key.
fn string_token_estimate(value: &Value, count_object_keys: bool) -> usize {
    match value {
        Value::String(text) => text_token_estimate(text),
        Value::Array(values) => values.iter().fold(0_usize, |total, value| {
            total.saturating_add(string_token_estimate(value, count_object_keys))
        }),
        Value::Object(values) => values.iter().fold(0_usize, |total, (key, value)| {
            let key_tokens = if count_object_keys { text_token_estimate(key) } else { 0 };
            total
                .saturating_add(key_tokens)
                .saturating_add(string_token_estimate(value, count_object_keys))
        }),
        Value::Null | Value::Bool(_) | Value::Number(_) => 0,
    }
}

/// Conservatively convert one string to the routing token estimate.
fn text_token_estimate(text: &str) -> usize {
    let ascii_bytes = text.bytes().filter(u8::is_ascii).count();
    let non_ascii_bytes = text.len().saturating_sub(ascii_bytes);
    ascii_bytes
        .div_ceil(ASCII_CHARS_PER_TOKEN)
        .saturating_add(non_ascii_bytes)
}

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

    fn config(max_input_tokens: usize) -> serde_yaml::Value {
        serde_yaml::from_str(&format!(
            "routes:\n  - model: small-model\n    max_input_tokens: {max_input_tokens}\n    overflow: large-context"
        ))
        .unwrap()
    }

    #[test]
    fn estimate_is_conservative_for_known_cl100k_samples() {
        let hello = serde_json::json!({"messages": [{"role": "user", "content": "hello world"}]});
        let pangram = serde_json::json!({
            "messages": [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]
        });
        let hello_estimate = estimate_prompt_tokens(&hello);
        let pangram_estimate = estimate_prompt_tokens(&pangram);

        assert!((2..=6).contains(&hello_estimate), "known cl100k count is 2");
        assert!((10..=22).contains(&pangram_estimate), "known cl100k count is 10");
    }

    #[test]
    fn estimate_supports_common_provider_prompt_shapes() {
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "abc"}]}],
            "system": "def",
            "instructions": "ghi",
            "input": "jkl",
            "prompt": "mno"
        });
        assert_eq!(estimate_prompt_tokens(&body), 9);
    }

    #[test]
    fn model_probe_uses_the_last_duplicate_model_value() {
        let probe = serde_json::from_slice::<ModelProbe<'_>>(
            br#"{"ignored":{"nested":[1,2,3]},"model":"old","model":"small-model"}"#,
        )
        .unwrap();

        assert_eq!(probe.0.as_deref(), Some("small-model"));
    }

    #[test]
    fn model_probe_treats_a_non_string_last_model_as_missing() {
        let probe =
            serde_json::from_slice::<ModelProbe<'_>>(br#"{"model":"small-model","model":{"name":"other"}}"#).unwrap();

        assert!(probe.0.is_none());
    }

    #[test]
    fn estimate_conservatively_charges_non_ascii_bytes() {
        let body = serde_json::json!({"prompt": "你好🙂"});
        assert_eq!(estimate_prompt_tokens(&body), 10);
    }

    #[test]
    fn estimate_counts_schema_property_names() {
        let long_property_name = "p".repeat(3_000);
        let body = serde_json::json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            (long_property_name): {"type": "string"}
                        }
                    }
                }
            }
        });

        assert!(
            estimate_prompt_tokens(&body) > 100,
            "a long schema property name must affect overflow routing"
        );
    }

    #[test]
    fn from_config_rejects_duplicate_models() {
        let yaml = serde_yaml::from_str(
            "routes:\n  - model: same\n    max_input_tokens: 10\n    overflow: large-a\n  - model: same\n    max_input_tokens: 20\n    overflow: large-b",
        )
        .unwrap();
        let error = PromptLengthToHeaderFilter::from_config(&yaml).err().unwrap();
        assert!(error.to_string().contains("duplicate model 'same'"));
    }

    #[test]
    fn from_config_rejects_zero_limit() {
        let error = PromptLengthToHeaderFilter::from_config(&config(0)).err().unwrap();
        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn from_config_rejects_unsafe_header() {
        let yaml = serde_yaml::from_str(
            "header: authorization\nroutes:\n  - model: small-model\n    max_input_tokens: 10\n    overflow: large",
        )
        .unwrap();
        let error = PromptLengthToHeaderFilter::from_config(&yaml).err().unwrap();
        assert!(error.to_string().contains("authorization"));
    }

    #[test]
    fn body_mode_uses_default_json_limit() {
        let filter = PromptLengthToHeaderFilter::from_config(&config(10)).unwrap();
        assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
        assert_eq!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer {
                max_bytes: Some(DEFAULT_JSON_BODY_MAX_BYTES)
            }
        );
    }

    #[tokio::test]
    async fn large_prompt_promotes_overflow_route() {
        let filter = PromptLengthToHeaderFilter::from_config(&config(3)).unwrap();
        let mut request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        request
            .headers
            .insert("x-prompt-route", http::HeaderValue::from_static("spoofed"));
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"model":"small-model","messages":[{"role":"user","content":"This prompt is too large"}]}"#,
        ));

        let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(
            context
                .request_headers_to_remove
                .iter()
                .any(|name| name == "x-prompt-route")
        );
        assert_eq!(
            context.extra_request_headers,
            vec![(Cow::Owned("x-prompt-route".to_owned()), "large-context".to_owned())]
        );
        assert_eq!(context.get_metadata(META_OVERFLOW_ROUTE), Some("large-context"));
    }

    #[tokio::test]
    async fn tool_and_structured_output_fields_trigger_overflow() {
        for field in [
            "tools",
            "functions",
            "response_format",
            "text",
            "output_config",
            "output_format",
        ] {
            let filter = PromptLengthToHeaderFilter::from_config(&config(2)).unwrap();
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
            let mut context = crate::test_utils::make_filter_context(&request);
            let mut body = Some(Bytes::from(format!(
                r#"{{"model":"small-model","{field}":{{"schema":"a deliberately large schema"}}}}"#
            )));

            drop(filter.on_request_body(&mut context, &mut body, true).await.unwrap());

            assert_eq!(
                context.get_metadata(META_OVERFLOW_ROUTE),
                Some("large-context"),
                "field {field} should contribute to the estimate"
            );
        }
    }

    #[tokio::test]
    async fn small_prompt_stays_on_default_route() {
        let filter = PromptLengthToHeaderFilter::from_config(&config(100)).unwrap();
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"model":"small-model","messages":[{"role":"user","content":"hello"}]}"#,
        ));

        drop(filter.on_request_body(&mut context, &mut body, true).await.unwrap());

        assert!(context.extra_request_headers.is_empty());
        assert_eq!(context.get_metadata(META_ESTIMATED_TOKENS), Some("4"));
        assert_eq!(context.get_metadata(META_OVERFLOW_ROUTE), None);
    }

    #[tokio::test]
    async fn unknown_model_removes_spoofed_header_without_promoting() {
        let filter = PromptLengthToHeaderFilter::from_config(&config(3)).unwrap();
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"model":"other-model","messages":[{"role":"user","content":"large prompt"}]}"#,
        ));

        drop(filter.on_request_body(&mut context, &mut body, true).await.unwrap());

        assert!(
            context
                .request_headers_to_remove
                .iter()
                .any(|name| name == "x-prompt-route")
        );
        assert!(context.extra_request_headers.is_empty());
    }

    #[tokio::test]
    async fn invalid_or_missing_models_continue_without_promoting() {
        for raw in [
            b"not json".as_slice(),
            b"[]".as_slice(),
            b"{}".as_slice(),
            br#"{"model":42,"prompt":"large"}"#.as_slice(),
        ] {
            let filter = PromptLengthToHeaderFilter::from_config(&config(1)).unwrap();
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
            let mut context = crate::test_utils::make_filter_context(&request);
            let mut body = Some(Bytes::copy_from_slice(raw));

            let action = filter.on_request_body(&mut context, &mut body, true).await.unwrap();

            assert!(matches!(action, FilterAction::Continue));
            assert!(context.extra_request_headers.is_empty());
            assert_eq!(context.get_metadata(META_ESTIMATED_TOKENS), None);
        }
    }

    #[tokio::test]
    async fn waits_for_end_of_stream() {
        let filter = PromptLengthToHeaderFilter::from_config(&config(1)).unwrap();
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(br#"{"model":"small-model","prompt":"large"}"#));

        drop(filter.on_request_body(&mut context, &mut body, false).await.unwrap());

        assert!(context.request_headers_to_remove.is_empty());
        assert!(context.extra_request_headers.is_empty());
    }
}
