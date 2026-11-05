// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Multimodal request detection for content-based routing.

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_ai_apis::promotion::parse_dedicated_promotion_header;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::Deserialize;
use serde_json::Value;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default proxy-owned fact header used by routing filters.
const DEFAULT_HEADER: &str = "X-Praxis-AI-Modality";
/// Metadata key carrying the detected content modality.
const META_MODALITY: &str = "multimodal_to_header.modality";

// -----------------------------------------------------------------------------
// Public Types
// -----------------------------------------------------------------------------

/// Detects image and audio content in message arrays and promotes a routing fact.
///
/// Detection covers typed OpenAI content parts (`image_url`, `input_image`,
/// `input_audio`, `audio_url`), image/audio data URLs, and base64 sources with
/// an image or audio media type. Text-only or malformed bodies do not publish a
/// modality fact.
///
/// The filter removes a client-supplied copy of its output header before
/// publishing `image`, `audio`, or `image_audio`. A following core `router` can
/// exact-match those values to keep multimodal content away from text-only
/// backends.
///
/// # YAML configuration
///
/// ```yaml
/// filter: multimodal_to_header
/// header: X-Praxis-AI-Modality
/// max_body_bytes: 1048576
/// ```
pub struct MultimodalToHeaderFilter {
    /// Header owned by this filter and replaced with the derived modality.
    header: HeaderName,
    /// Maximum request body buffered for content detection.
    max_body_bytes: usize,
}

impl MultimodalToHeaderFilter {
    /// Construct the filter from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe header or invalid body limit.
    ///
    /// # Example
    ///
    /// ```
    /// use praxis_ai_filters::MultimodalToHeaderFilter;
    ///
    /// let filter = MultimodalToHeaderFilter::from_config(&serde_yaml::Value::Null).unwrap();
    /// assert_eq!(filter.name(), "multimodal_to_header");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: MultimodalToHeaderConfig = parse_filter_config("multimodal_to_header", config)?;
        validate_max_body_bytes("multimodal_to_header", config.max_body_bytes)?;
        let header =
            parse_dedicated_promotion_header("multimodal_to_header", "header", &config.header, &[DEFAULT_HEADER])?;

        Ok(Box::new(Self {
            header,
            max_body_bytes: config.max_body_bytes,
        }))
    }
}

#[async_trait]
impl HttpFilter for MultimodalToHeaderFilter {
    fn name(&self) -> &'static str {
        "multimodal_to_header"
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
        ctx.filter_metadata.remove(META_MODALITY);

        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(value) = serde_json::from_slice::<Value>(raw) else {
            return Ok(FilterAction::Continue);
        };
        let Some(modality) = detect_content_modality(&value) else {
            return Ok(FilterAction::Continue);
        };

        let modality = modality.as_str();
        ctx.extra_request_headers
            .push((Cow::Owned(self.header.as_str().to_owned()), modality.to_owned()));
        ctx.set_metadata(META_MODALITY, modality);

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Private Types
// -----------------------------------------------------------------------------

/// Deserialized filter configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MultimodalToHeaderConfig {
    /// Header that receives the detected modality.
    #[serde(default = "default_header")]
    header: String,
    /// Maximum JSON request body buffered for detection.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// A bounded routing value derived from detected request content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContentModality {
    /// Audio content only.
    Audio,
    /// Image content only.
    Image,
    /// Both image and audio content.
    ImageAudio,
}

impl ContentModality {
    /// Return the stable value published to headers and metadata.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::Image => "image",
            Self::ImageAudio => "image_audio",
        }
    }
}

/// Image and audio signals accumulated while walking message content.
#[derive(Default)]
struct DetectedModalities {
    /// Whether audio content was observed.
    audio: bool,
    /// Whether image content was observed.
    image: bool,
}

impl DetectedModalities {
    /// Return `true` once no additional modality can change the result.
    const fn is_complete(&self) -> bool {
        self.audio && self.image
    }

    /// Convert accumulated signals into the public routing value.
    const fn modality(&self) -> Option<ContentModality> {
        match (self.image, self.audio) {
            (false, false) => None,
            (false, true) => Some(ContentModality::Audio),
            (true, false) => Some(ContentModality::Image),
            (true, true) => Some(ContentModality::ImageAudio),
        }
    }

    /// Record a media type when it identifies image or audio content.
    fn observe_media_type(&mut self, media_type: &str) {
        if starts_with_ascii_case_insensitive(media_type, "image/") {
            self.image = true;
        } else if starts_with_ascii_case_insensitive(media_type, "audio/") {
            self.audio = true;
        }
    }

    /// Record a typed content-part discriminator.
    fn observe_part_type(&mut self, part_type: &str) {
        if matches!(part_type, "image" | "image_url" | "input_image") {
            self.image = true;
        } else if matches!(part_type, "audio" | "audio_url" | "input_audio") {
            self.audio = true;
        }
    }

    /// Record an image/audio data URL embedded in content.
    fn observe_string(&mut self, value: &str) {
        if starts_with_ascii_case_insensitive(value, "data:image/") {
            self.image = true;
        } else if starts_with_ascii_case_insensitive(value, "data:audio/") {
            self.audio = true;
        }
    }
}

// -----------------------------------------------------------------------------
// Private Functions
// -----------------------------------------------------------------------------

/// Default proxy-owned modality header.
fn default_header() -> String {
    DEFAULT_HEADER.to_owned()
}

/// Default request body limit shared by JSON body filters.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Detect image and audio signals under top-level message content fields.
fn detect_content_modality(body: &Value) -> Option<ContentModality> {
    let messages = body.get("messages")?.as_array()?;
    let mut detected = DetectedModalities::default();

    for message in messages {
        if let Some(content) = message.get("content") {
            detect_in_content(content, &mut detected);
        }
        if detected.is_complete() {
            break;
        }
    }

    detected.modality()
}

/// Walk one message content value and accumulate media signals.
fn detect_in_content(value: &Value, detected: &mut DetectedModalities) {
    match value {
        Value::Array(values) => {
            for value in values {
                detect_in_content(value, detected);
                if detected.is_complete() {
                    break;
                }
            }
        },
        Value::Object(fields) => detect_in_object(fields, detected),
        Value::Bool(_) | Value::Null | Value::Number(_) | Value::String(_) => {},
    }
}

/// Inspect one content-part object and recurse into its nested values.
fn detect_in_object(fields: &serde_json::Map<String, Value>, detected: &mut DetectedModalities) {
    if let Some(part_type) = fields.get("type").and_then(Value::as_str) {
        detected.observe_part_type(part_type);
    }
    if fields.get("image_url").is_some_and(|value| !value.is_null()) {
        detected.image = true;
    }
    if fields
        .get("audio_url")
        .or_else(|| fields.get("input_audio"))
        .is_some_and(|value| !value.is_null())
    {
        detected.audio = true;
    }
    if let Some(media_type) = fields.get("media_type").and_then(Value::as_str) {
        detected.observe_media_type(media_type);
    }
    for field in ["data", "url"] {
        if let Some(value) = fields.get(field).and_then(Value::as_str) {
            detected.observe_string(value);
        }
    }
    for value in fields.values() {
        detect_in_content(value, detected);
        if detected.is_complete() {
            break;
        }
    }
}

/// Compare an ASCII prefix without allocating a lowercased media payload.
fn starts_with_ascii_case_insensitive(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_url_content_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]
            }]
        });

        assert_eq!(detect_content_modality(&body), Some(ContentModality::Image));
    }

    #[test]
    fn image_data_url_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
            }]
        });

        assert_eq!(detect_content_modality(&body), Some(ContentModality::Image));
    }

    #[test]
    fn input_audio_base64_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]
            }]
        });

        assert_eq!(detect_content_modality(&body), Some(ContentModality::Audio));
    }

    #[test]
    fn audio_url_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"audio_url": {"url": "https://example.test/audio.wav"}}]
            }]
        });

        assert_eq!(detect_content_modality(&body), Some(ContentModality::Audio));
    }

    #[test]
    fn image_and_audio_are_detected_together() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}
                ]
            }]
        });

        assert_eq!(detect_content_modality(&body), Some(ContentModality::ImageAudio));
    }

    #[test]
    fn base64_source_media_type_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ"}
                }]
            }]
        });

        assert_eq!(detect_content_modality(&body), Some(ContentModality::Image));
    }

    #[test]
    fn text_only_content_has_no_modality() {
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "describe a sunrise"}]
        });

        assert_eq!(detect_content_modality(&body), None);
    }

    #[test]
    fn data_url_mentioned_in_text_has_no_modality() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Explain data:image/png;base64,iVBORw=="}]
            }]
        });

        assert_eq!(detect_content_modality(&body), None);
    }

    #[test]
    fn non_message_media_fields_are_ignored() {
        let body = serde_json::json!({
            "image_url": "https://example.test/outside.png",
            "messages": [{"role": "user", "content": "text only"}]
        });

        assert_eq!(detect_content_modality(&body), None);
    }

    #[test]
    fn from_config_rejects_unsafe_header() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = serde_yaml::from_str("header: authorization")?;
        let result = MultimodalToHeaderFilter::from_config(&config);

        assert!(result.is_err(), "credential headers must not carry routing facts");
        Ok(())
    }

    #[test]
    fn from_config_accepts_custom_header() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = serde_yaml::from_str("header: X-Content-Modality")?;
        let filter = MultimodalToHeaderFilter::from_config(&config)?;

        assert_eq!(filter.name(), "multimodal_to_header");
        Ok(())
    }

    #[test]
    fn from_config_rejects_zero_body_limit() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = serde_yaml::from_str("max_body_bytes: 0")?;
        let result = MultimodalToHeaderFilter::from_config(&config);

        assert!(result.is_err(), "zero max_body_bytes must be rejected");
        Ok(())
    }

    #[test]
    fn body_mode_uses_default_json_limit() -> Result<(), FilterError> {
        let filter = MultimodalToHeaderFilter::from_config(&serde_yaml::Value::Null)?;

        assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
        assert_eq!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer {
                max_bytes: Some(DEFAULT_JSON_BODY_MAX_BYTES)
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn image_audio_fact_replaces_spoofed_header() -> Result<(), FilterError> {
        let filter = MultimodalToHeaderFilter::from_config(&serde_yaml::Value::Null)?;
        let mut request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        request
            .headers
            .insert("x-praxis-ai-modality", http::HeaderValue::from_static("image"));
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/a.png"}},{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}]}]}"#,
        ));

        let action = filter.on_request_body(&mut context, &mut body, true).await?;

        assert!(matches!(action, FilterAction::Continue));
        assert!(
            context
                .request_headers_to_remove
                .iter()
                .any(|name| name == "x-praxis-ai-modality"),
            "the client-supplied routing fact must be removed"
        );
        assert_eq!(
            context.extra_request_headers,
            vec![(Cow::Owned("x-praxis-ai-modality".to_owned()), "image_audio".to_owned())]
        );
        assert_eq!(context.get_metadata(META_MODALITY), Some("image_audio"));
        Ok(())
    }

    #[tokio::test]
    async fn text_only_body_removes_spoofed_header_without_promotion() -> Result<(), FilterError> {
        let filter = MultimodalToHeaderFilter::from_config(&serde_yaml::Value::Null)?;
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"messages":[{"role":"user","content":"text only"}]}"#,
        ));

        let action = filter.on_request_body(&mut context, &mut body, true).await?;

        assert!(matches!(action, FilterAction::Continue));
        assert!(context.extra_request_headers.is_empty());
        assert_eq!(context.get_metadata(META_MODALITY), None);
        assert!(
            context
                .request_headers_to_remove
                .iter()
                .any(|name| name == "x-praxis-ai-modality"),
            "the owned routing header must always be removed"
        );
        Ok(())
    }

    #[tokio::test]
    async fn waits_for_end_of_stream() -> Result<(), FilterError> {
        let filter = MultimodalToHeaderFilter::from_config(&serde_yaml::Value::Null)?;
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(
            br#"{"messages":[{"content":[{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}]}]}"#,
        ));

        let action = filter.on_request_body(&mut context, &mut body, false).await?;

        assert!(matches!(action, FilterAction::Continue));
        assert!(context.request_headers_to_remove.is_empty());
        assert!(context.extra_request_headers.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn malformed_body_removes_owned_header_without_promotion() -> Result<(), FilterError> {
        let filter = MultimodalToHeaderFilter::from_config(&serde_yaml::Value::Null)?;
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut context = crate::test_utils::make_filter_context(&request);
        let mut body = Some(Bytes::from_static(b"not json"));

        let action = filter.on_request_body(&mut context, &mut body, true).await?;

        assert!(matches!(action, FilterAction::Continue));
        assert!(context.extra_request_headers.is_empty());
        assert!(
            context
                .request_headers_to_remove
                .iter()
                .any(|name| name == "x-praxis-ai-modality"),
            "malformed bodies must not preserve a staged modality fact"
        );
        Ok(())
    }
}
