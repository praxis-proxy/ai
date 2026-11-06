// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Multimodal request detection for content-based routing.

use std::{borrow::Cow, fmt, marker::PhantomData, ops::Deref};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_ai_apis::promotion::parse_dedicated_promotion_header;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::{
    Deserialize, Deserializer,
    de::{self, Visitor},
};
use serde_json::value::RawValue;

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
/// Bodies larger than `max_body_bytes` are rejected with HTTP 413 before
/// routing. The default is 10 MiB. Operators should account for base64
/// expansion when sizing this limit for image and audio payloads.
///
/// # YAML configuration
///
/// ```yaml
/// filter: multimodal_to_header
/// header: X-Praxis-AI-Modality
/// # Base64 expands media payloads; size this for the largest accepted request.
/// max_body_bytes: 16777216 # 16 MiB
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
    /// Return the registry name for this filter.
    fn name(&self) -> &'static str {
        "multimodal_to_header"
    }

    /// Continue to body processing, where the routing fact is derived.
    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    /// Declare read-only access because classification never mutates the body.
    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// Buffer at most the configured number of bytes for classification.
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    /// Derive and publish the modality after the complete body is available.
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
        let Some(modality) = detect_content_modality(raw) else {
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

/// A JSON string that borrows unescaped input and owns only decoded escapes.
enum JsonString<'a> {
    /// Slice borrowed directly from the buffered request body.
    Borrowed(&'a str),
    /// Decoded string required when the JSON source contains escapes.
    Owned(String),
}

impl Deref for JsonString<'_> {
    type Target = str;

    /// Return the decoded string regardless of its ownership mode.
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<'de: 'a, 'a> Deserialize<'de> for JsonString<'a> {
    /// Borrow unescaped strings and allocate only when decoding is required.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(JsonStringVisitor(PhantomData))
    }
}

/// Serde visitor that preserves borrowed JSON string slices.
struct JsonStringVisitor<'a>(PhantomData<&'a str>);

impl<'de: 'a, 'a> Visitor<'de> for JsonStringVisitor<'a> {
    type Value = JsonString<'a>;

    /// Describe the accepted JSON value.
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON string")
    }

    /// Preserve a string borrowed from the request body.
    fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonString::Borrowed(v))
    }

    /// Own a transient decoded string supplied by the deserializer.
    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonString::Owned(v.to_owned()))
    }

    /// Retain an already-owned decoded string without another copy.
    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonString::Owned(v))
    }
}

/// Borrowed request fields needed for message-content classification.
#[derive(Deserialize)]
struct MultimodalRequest<'a> {
    /// OpenAI-style conversation messages.
    #[serde(borrow, default)]
    messages: Vec<Message<'a>>,
}

/// Borrowed message fields needed for content classification.
#[derive(Deserialize)]
struct Message<'a> {
    /// Text, a single content part, or an array of content parts.
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
}

/// Borrowed fields from a structured message content part.
#[derive(Deserialize)]
struct ContentPart<'a> {
    /// Content-part discriminator.
    #[serde(borrow, default, rename = "type")]
    part_type: Option<JsonString<'a>>,
    /// OpenAI image source.
    #[serde(borrow, default)]
    image_url: Option<&'a RawValue>,
    /// URL-backed audio source.
    #[serde(borrow, default)]
    audio_url: Option<&'a RawValue>,
    /// OpenAI inline audio source.
    #[serde(borrow, default)]
    input_audio: Option<&'a RawValue>,
    /// Anthropic-style inline media source.
    #[serde(borrow, default)]
    source: Option<&'a RawValue>,
    /// File-backed image source; the value itself is irrelevant for routing.
    #[serde(borrow, default)]
    file_id: Option<&'a RawValue>,
    /// Inline media type when it appears directly on the part.
    #[serde(borrow, default)]
    media_type: Option<JsonString<'a>>,
    /// Inline data or data URL when it appears directly on the part.
    #[serde(borrow, default)]
    data: Option<JsonString<'a>>,
    /// Remote or data URL when it appears directly on the part.
    #[serde(borrow, default)]
    url: Option<JsonString<'a>>,
}

/// Borrowed media fields nested beneath a content part.
#[derive(Deserialize)]
struct MediaSourceFields<'a> {
    /// MIME type for an inline payload.
    #[serde(borrow, default)]
    media_type: Option<JsonString<'a>>,
    /// Inline payload or data URL.
    #[serde(borrow, default)]
    data: Option<JsonString<'a>>,
    /// Remote or data URL.
    #[serde(borrow, default)]
    url: Option<JsonString<'a>>,
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

    /// Record a typed discriminator only when its matching source exists.
    fn observe_part_type(&mut self, part_type: &str, has_image_source: bool, has_audio_source: bool) {
        if has_image_source && matches!(part_type, "image" | "image_url" | "input_image") {
            self.image = true;
        } else if has_audio_source && matches!(part_type, "audio" | "audio_url" | "input_audio") {
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

impl ContentPart<'_> {
    /// Accumulate modality signals from one structured content part.
    fn detect(&self, detected: &mut DetectedModalities) {
        let part_type = self.part_type.as_deref();
        let has_image_source = self.image_url.is_some()
            || (part_type == Some("input_image") && self.file_id.is_some())
            || (part_type == Some("image") && self.source.is_some());
        let has_audio_source = self.audio_url.is_some()
            || self.input_audio.is_some()
            || (part_type == Some("audio") && self.source.is_some());

        if let Some(part_type) = part_type {
            detected.observe_part_type(part_type, has_image_source, has_audio_source);
        }
        if self.image_url.is_some() {
            detected.image = true;
        }
        if self.audio_url.is_some() || self.input_audio.is_some() {
            detected.audio = true;
        }

        self.detect_source_details(detected);
    }

    /// Inspect nested source objects and direct media metadata.
    fn detect_source_details(&self, detected: &mut DetectedModalities) {
        for source in [
            self.image_url.as_ref(),
            self.audio_url.as_ref(),
            self.input_audio.as_ref(),
            self.source.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            detect_media_source(source, detected);
        }
        if let Some(media_type) = self.media_type.as_deref() {
            detected.observe_media_type(media_type);
        }
        for value in [self.data.as_deref(), self.url.as_deref()].into_iter().flatten() {
            detected.observe_string(value);
        }
    }
}

impl MediaSourceFields<'_> {
    /// Accumulate MIME and data-URL signals without copying payload strings.
    fn detect(&self, detected: &mut DetectedModalities) {
        if let Some(media_type) = self.media_type.as_deref() {
            detected.observe_media_type(media_type);
        }
        for value in [self.data.as_deref(), self.url.as_deref()].into_iter().flatten() {
            detected.observe_string(value);
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
fn detect_content_modality(body: &[u8]) -> Option<ContentModality> {
    let request = serde_json::from_slice::<MultimodalRequest<'_>>(body).ok()?;
    let mut detected = DetectedModalities::default();

    for message in request.messages {
        if let Some(content) = message.content {
            detect_message_content(content, &mut detected);
        }
        if detected.is_complete() {
            break;
        }
    }

    detected.modality()
}

/// Inspect a message content object or array through borrowed raw JSON slices.
fn detect_message_content(content: &RawValue, detected: &mut DetectedModalities) {
    let json = content.get().trim_start();
    if json.starts_with('[') {
        let Ok(parts) = serde_json::from_str::<Vec<&RawValue>>(json) else {
            return;
        };
        for part in parts {
            detect_content_part(part, detected);
            if detected.is_complete() {
                break;
            }
        }
    } else if json.starts_with('{') {
        detect_content_part(content, detected);
    }
}

/// Inspect one structured content part without materializing payload strings.
fn detect_content_part(part: &RawValue, detected: &mut DetectedModalities) {
    if let Ok(part) = serde_json::from_str::<ContentPart<'_>>(part.get()) {
        part.detect(detected);
    }
}

/// Inspect a direct string or structured media source through borrowed JSON.
fn detect_media_source(source: &RawValue, detected: &mut DetectedModalities) {
    let json = source.get().trim_start();
    if json.starts_with('"') {
        if let Ok(value) = serde_json::from_str::<&str>(json) {
            detected.observe_string(value);
        } else if let Ok(value) = serde_json::from_str::<String>(json) {
            detected.observe_string(&value);
        }
    } else if json.starts_with('{')
        && let Ok(fields) = serde_json::from_str::<MediaSourceFields<'_>>(json)
    {
        fields.detect(detected);
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
#[expect(clippy::expect_used, reason = "test fixture construction")]
mod tests {
    use super::*;

    /// Serialize a unit-test fixture and classify its borrowed JSON bytes.
    fn detect_json(body: &serde_json::Value) -> Option<ContentModality> {
        let body = serde_json::to_vec(body).expect("test fixture should serialize");
        detect_content_modality(&body)
    }

    /// Detect an OpenAI image URL content part.
    #[test]
    fn image_url_content_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]
            }]
        });

        assert_eq!(detect_json(&body), Some(ContentModality::Image));
    }

    /// Detect a data URL nested beneath `image_url`.
    #[test]
    fn image_data_url_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
            }]
        });

        assert_eq!(detect_json(&body), Some(ContentModality::Image));
    }

    /// Detect an OpenAI inline audio content part.
    #[test]
    fn input_audio_base64_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]
            }]
        });

        assert_eq!(detect_json(&body), Some(ContentModality::Audio));
    }

    /// Detect a URL-backed audio content part.
    #[test]
    fn audio_url_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"audio_url": {"url": "https://example.test/audio.wav"}}]
            }]
        });

        assert_eq!(detect_json(&body), Some(ContentModality::Audio));
    }

    /// Combine image and audio signals into the bounded routing value.
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

        assert_eq!(detect_json(&body), Some(ContentModality::ImageAudio));
    }

    /// Detect an Anthropic-style base64 source from its media type.
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

        assert_eq!(detect_json(&body), Some(ContentModality::Image));
    }

    /// Preserve file-backed `input_image` classification.
    #[test]
    fn input_image_file_id_is_detected() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file_123"}]
            }]
        });

        assert_eq!(detect_json(&body), Some(ContentModality::Image));
    }

    /// Ignore typed parts whose corresponding source is absent or null.
    #[test]
    fn discriminator_without_source_has_no_modality() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url"},
                    {"type": "input_image", "file_id": null},
                    {"type": "input_audio", "input_audio": null}
                ]
            }]
        });

        assert_eq!(detect_json(&body), None);
    }

    /// Keep an unescaped media payload borrowed from the buffered request.
    #[test]
    fn media_payload_is_borrowed() {
        let body = br#"{"messages":[{"content":[{"type":"input_audio","input_audio":{"data":"UklGRg=="}}]}]}"#;
        let request = serde_json::from_slice::<MultimodalRequest<'_>>(body).expect("fixture should parse");
        let content = request
            .messages
            .first()
            .and_then(|message| message.content)
            .expect("fixture should contain message content");
        let parts = serde_json::from_str::<Vec<&RawValue>>(content.get()).expect("content parts should parse");
        let part = parts.first().expect("fixture should contain one part");
        let part = serde_json::from_str::<ContentPart<'_>>(part.get()).expect("content part should parse");
        let source = part.input_audio.expect("fixture should contain an audio source");
        let source = serde_json::from_str::<MediaSourceFields<'_>>(source.get()).expect("audio source should parse");

        assert!(
            matches!(source.data, Some(JsonString::Borrowed("UklGRg=="))),
            "unescaped base64 data should borrow from the buffered body"
        );
    }

    /// Decode escaped media strings without changing their classification.
    #[test]
    fn escaped_data_url_is_detected() {
        let body = br#"{"messages":[{"content":[{"image_url":{"url":"data:image\/png;base64,iVBORw=="}}]}]}"#;

        assert_eq!(detect_content_modality(body), Some(ContentModality::Image));
    }

    /// Leave plain text requests unclassified.
    #[test]
    fn text_only_content_has_no_modality() {
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "describe a sunrise"}]
        });

        assert_eq!(detect_json(&body), None);
    }

    /// Do not treat a data URL mentioned in text as an actual media source.
    #[test]
    fn data_url_mentioned_in_text_has_no_modality() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Explain data:image/png;base64,iVBORw=="}]
            }]
        });

        assert_eq!(detect_json(&body), None);
    }

    /// Ignore media-like fields outside message content.
    #[test]
    fn non_message_media_fields_are_ignored() {
        let body = serde_json::json!({
            "image_url": "https://example.test/outside.png",
            "messages": [{"role": "user", "content": "text only"}]
        });

        assert_eq!(detect_json(&body), None);
    }

    /// Reject credential-bearing output headers.
    #[test]
    fn from_config_rejects_unsafe_header() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = serde_yaml::from_str("header: authorization")?;
        let result = MultimodalToHeaderFilter::from_config(&config);

        assert!(result.is_err(), "credential headers must not carry routing facts");
        Ok(())
    }

    /// Accept a non-reserved custom output header.
    #[test]
    fn from_config_accepts_custom_header() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = serde_yaml::from_str("header: X-Content-Modality")?;
        let filter = MultimodalToHeaderFilter::from_config(&config)?;

        assert_eq!(filter.name(), "multimodal_to_header");
        Ok(())
    }

    /// Reject a zero-byte body budget.
    #[test]
    fn from_config_rejects_zero_body_limit() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = serde_yaml::from_str("max_body_bytes: 0")?;
        let result = MultimodalToHeaderFilter::from_config(&config);

        assert!(result.is_err(), "zero max_body_bytes must be rejected");
        Ok(())
    }

    /// Use the shared JSON body limit by default.
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

    /// Replace a spoofed value with the modality derived from content.
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

    /// Remove a spoofed value without promoting text-only content.
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

    /// Defer classification until the complete buffered body is available.
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

    /// Remove stale routing facts even when the JSON body is malformed.
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
