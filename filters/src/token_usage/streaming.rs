// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Streaming token extraction from individual SSE events.
//!
//! Handles providers that spread token counts across multiple SSE
//! events rather than including complete usage in a single event.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// Anthropic Streaming
// -----------------------------------------------------------------------------

/// Anthropic `message_start` event with nested usage under `message`.
#[derive(Deserialize)]
struct AnthropicMessageStart {
    /// Nested message object containing usage.
    message: Option<AnthropicMessageStartMessage>,

    /// Event type discriminator.
    #[serde(rename = "type")]
    event_type: Option<String>,
}

/// Inner message object from `message_start`.
#[derive(Deserialize)]
struct AnthropicMessageStartMessage {
    /// Token usage for the input.
    usage: Option<AnthropicStartUsage>,
}

/// Usage object inside `message_start.message.usage`.
#[derive(Deserialize)]
struct AnthropicStartUsage {
    /// Tokens in the input prompt.
    input_tokens: u64,

    /// Tokens written to cache (prompt caching).
    cache_creation_input_tokens: Option<u64>,

    /// Tokens read from cache (prompt caching).
    cache_read_input_tokens: Option<u64>,
}

/// Anthropic `message_delta` event with usage at root level.
#[derive(Deserialize)]
struct AnthropicMessageDelta {
    /// Event type discriminator.
    #[serde(rename = "type")]
    event_type: Option<String>,

    /// Token usage for the output.
    usage: Option<AnthropicDeltaUsage>,
}

/// Usage object inside `message_delta.usage`.
#[derive(Deserialize)]
struct AnthropicDeltaUsage {
    /// Tokens in the output completion.
    output_tokens: u64,
}

/// Parses Anthropic streaming events for partial token counts.
pub(super) fn parse_anthropic_event(data: &[u8]) -> (Option<u64>, Option<u64>) {
    if let Ok(start) = serde_json::from_slice::<AnthropicMessageStart>(data)
        && start.event_type.as_deref() == Some("message_start")
        && let Some(message) = start.message
        && let Some(usage) = message.usage
    {
        let actual_input = usage
            .input_tokens
            .saturating_add(usage.cache_creation_input_tokens.unwrap_or(0))
            .saturating_add(usage.cache_read_input_tokens.unwrap_or(0));
        return (Some(actual_input), None);
    }

    if let Ok(delta) = serde_json::from_slice::<AnthropicMessageDelta>(data)
        && delta.event_type.as_deref() == Some("message_delta")
        && let Some(usage) = delta.usage
    {
        return (None, Some(usage.output_tokens));
    }

    (None, None)
}

// -----------------------------------------------------------------------------
// Bedrock ConverseStream
// -----------------------------------------------------------------------------

/// Bedrock `ConverseStream` metadata event.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockStreamMetadata {
    /// Token usage metadata from the stream.
    metadata: Option<BedrockStreamMetadataInner>,
}

/// Inner metadata object containing usage.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockStreamMetadataInner {
    /// Token usage statistics.
    usage: Option<BedrockStreamUsage>,
}

/// Bedrock streaming usage object.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockStreamUsage {
    /// Tokens in the input.
    input_tokens: u64,

    /// Tokens in the output.
    output_tokens: u64,
}

/// Parses Bedrock `ConverseStream` metadata events for token counts.
pub(super) fn parse_bedrock_event(data: &[u8]) -> (Option<u64>, Option<u64>) {
    let Some(meta) = serde_json::from_slice::<BedrockStreamMetadata>(data).ok() else {
        return (None, None);
    };
    let Some(usage) = meta.metadata.and_then(|m| m.usage) else {
        return (None, None);
    };
    (Some(usage.input_tokens), Some(usage.output_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Anthropic: Malformed / Null
    // -------------------------------------------------------------------------

    #[test]
    fn anthropic_missing_type_field_returns_none() {
        let event = br#"{"message":{"usage":{"input_tokens":25}}}"#;
        assert_eq!(parse_anthropic_event(event), (None, None));
    }

    #[test]
    fn anthropic_wrong_type_with_usage_returns_none() {
        let event = br#"{"type":"message_stop","message":{"usage":{"input_tokens":25}}}"#;
        assert_eq!(parse_anthropic_event(event), (None, None));
    }

    #[test]
    fn anthropic_message_start_with_null_message() {
        let event = br#"{"type":"message_start","message":null}"#;
        assert_eq!(parse_anthropic_event(event), (None, None));
    }

    #[test]
    fn anthropic_message_start_with_null_usage() {
        let event = br#"{"type":"message_start","message":{"usage":null}}"#;
        assert_eq!(parse_anthropic_event(event), (None, None));
    }

    #[test]
    fn anthropic_invalid_json_returns_none() {
        assert_eq!(parse_anthropic_event(b"not json"), (None, None));
    }

    #[test]
    fn anthropic_empty_returns_none() {
        assert_eq!(parse_anthropic_event(b""), (None, None));
    }

    // -------------------------------------------------------------------------
    // Bedrock: Malformed / Null
    // -------------------------------------------------------------------------

    #[test]
    fn bedrock_null_metadata() {
        let event = br#"{"metadata":null}"#;
        assert_eq!(parse_bedrock_event(event), (None, None));
    }

    #[test]
    fn bedrock_null_usage_inside_metadata() {
        let event = br#"{"metadata":{"usage":null}}"#;
        assert_eq!(parse_bedrock_event(event), (None, None));
    }

    #[test]
    fn bedrock_invalid_json_returns_none() {
        assert_eq!(parse_bedrock_event(b"not json"), (None, None));
    }

    #[test]
    fn bedrock_empty_returns_none() {
        assert_eq!(parse_bedrock_event(b""), (None, None));
    }
}
