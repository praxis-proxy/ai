// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming token extraction from individual SSE events.
//!
//! Handles providers that spread token counts across multiple SSE
//! events rather than including complete usage in a single event.

use serde::Deserialize;

use super::StreamingTokens;

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

    /// Breakdown of the output tokens (extended thinking). Present only on
    /// the terminal `message_delta`.
    output_tokens_details: Option<AnthropicOutputTokensDetails>,
}

/// Anthropic output token breakdown.
#[derive(Deserialize)]
struct AnthropicOutputTokensDetails {
    /// Thinking tokens, already counted in `output_tokens`.
    thinking_tokens: Option<u64>,
}

/// Parses Anthropic streaming events for partial token counts.
///
/// Input and cache counts arrive in `message_start`; output and thinking
/// counts arrive in `message_delta`. Cache counts break down the input;
/// thinking tokens break down the output.
pub(super) fn parse_anthropic_event(data: &[u8]) -> StreamingTokens {
    if let Some(tokens) = parse_anthropic_message_start(data) {
        return tokens;
    }
    if let Some(tokens) = parse_anthropic_message_delta(data) {
        return tokens;
    }
    StreamingTokens::default()
}

/// Input and cache counts from a `message_start` event.
fn parse_anthropic_message_start(data: &[u8]) -> Option<StreamingTokens> {
    let start = serde_json::from_slice::<AnthropicMessageStart>(data).ok()?;
    if start.event_type.as_deref() != Some("message_start") {
        return None;
    }
    let usage = start.message?.usage?;
    let cache_write = usage.cache_creation_input_tokens;
    let cache_read = usage.cache_read_input_tokens;
    let actual_input = usage
        .input_tokens
        .saturating_add(cache_write.unwrap_or(0))
        .saturating_add(cache_read.unwrap_or(0));
    Some(StreamingTokens {
        input: Some(actual_input),
        cache_read,
        cache_write,
        ..StreamingTokens::default()
    })
}

/// Output and thinking counts from a `message_delta` event.
fn parse_anthropic_message_delta(data: &[u8]) -> Option<StreamingTokens> {
    let delta = serde_json::from_slice::<AnthropicMessageDelta>(data).ok()?;
    if delta.event_type.as_deref() != Some("message_delta") {
        return None;
    }
    let usage = delta.usage?;
    Some(StreamingTokens {
        output: Some(usage.output_tokens),
        reasoning: usage.output_tokens_details.and_then(|details| details.thinking_tokens),
        ..StreamingTokens::default()
    })
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
///
/// The stream's metadata event carries no cache breakdown, so none is reported.
pub(super) fn parse_bedrock_event(data: &[u8]) -> StreamingTokens {
    let Some(meta) = serde_json::from_slice::<BedrockStreamMetadata>(data).ok() else {
        return StreamingTokens::default();
    };
    let Some(usage) = meta.metadata.and_then(|m| m.usage) else {
        return StreamingTokens::default();
    };
    StreamingTokens {
        input: Some(usage.input_tokens),
        output: Some(usage.output_tokens),
        ..StreamingTokens::default()
    }
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
        assert_eq!(parse_anthropic_event(event), StreamingTokens::default());
    }

    #[test]
    fn anthropic_wrong_type_with_usage_returns_none() {
        let event = br#"{"type":"message_stop","message":{"usage":{"input_tokens":25}}}"#;
        assert_eq!(parse_anthropic_event(event), StreamingTokens::default());
    }

    #[test]
    fn anthropic_message_start_with_null_message() {
        let event = br#"{"type":"message_start","message":null}"#;
        assert_eq!(parse_anthropic_event(event), StreamingTokens::default());
    }

    #[test]
    fn anthropic_message_start_with_null_usage() {
        let event = br#"{"type":"message_start","message":{"usage":null}}"#;
        assert_eq!(parse_anthropic_event(event), StreamingTokens::default());
    }

    #[test]
    fn anthropic_invalid_json_returns_none() {
        assert_eq!(parse_anthropic_event(b"not json"), StreamingTokens::default());
    }

    #[test]
    fn anthropic_empty_returns_none() {
        assert_eq!(parse_anthropic_event(b""), StreamingTokens::default());
    }

    // -------------------------------------------------------------------------
    // Bedrock: Malformed / Null
    // -------------------------------------------------------------------------

    #[test]
    fn bedrock_null_metadata() {
        let event = br#"{"metadata":null}"#;
        assert_eq!(parse_bedrock_event(event), StreamingTokens::default());
    }

    #[test]
    fn bedrock_null_usage_inside_metadata() {
        let event = br#"{"metadata":{"usage":null}}"#;
        assert_eq!(parse_bedrock_event(event), StreamingTokens::default());
    }

    #[test]
    fn bedrock_invalid_json_returns_none() {
        assert_eq!(parse_bedrock_event(b"not json"), StreamingTokens::default());
    }

    #[test]
    fn bedrock_empty_returns_none() {
        assert_eq!(parse_bedrock_event(b""), StreamingTokens::default());
    }

    // -------------------------------------------------------------------------
    // Prompt cache breakdown
    // -------------------------------------------------------------------------

    #[test]
    fn anthropic_message_start_reports_cache_breakdown() {
        let event = br#"{"type":"message_start","message":{"usage":{
            "input_tokens":10,
            "cache_creation_input_tokens":200,
            "cache_read_input_tokens":3000
        }}}"#;

        assert_eq!(
            parse_anthropic_event(event),
            StreamingTokens {
                input: Some(3210),
                output: None,
                cache_read: Some(3000),
                cache_write: Some(200),
                reasoning: None,
            }
        );
    }

    #[test]
    fn anthropic_message_start_without_cache_reports_absent_counts() {
        let event = br#"{"type":"message_start","message":{"usage":{"input_tokens":25}}}"#;

        assert_eq!(
            parse_anthropic_event(event),
            StreamingTokens {
                input: Some(25),
                output: None,
                cache_read: None,
                cache_write: None,
                reasoning: None,
            },
            "omitted cache fields stay absent instead of accumulating a zero"
        );
    }

    #[test]
    fn anthropic_message_start_with_zero_cache_reports_zero() {
        let event = br#"{"type":"message_start","message":{"usage":{"input_tokens":25,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}"#;

        assert_eq!(
            parse_anthropic_event(event),
            StreamingTokens {
                input: Some(25),
                output: None,
                cache_read: Some(0),
                cache_write: Some(0),
                reasoning: None,
            },
            "an explicit zero is a reported cache miss, distinct from an absent count"
        );
    }

    #[test]
    fn anthropic_message_delta_reports_no_cache() {
        let event = br#"{"type":"message_delta","usage":{"output_tokens":42}}"#;

        assert_eq!(
            parse_anthropic_event(event),
            StreamingTokens {
                input: None,
                output: Some(42),
                cache_read: None,
                cache_write: None,
                reasoning: None,
            }
        );
    }

    #[test]
    fn bedrock_metadata_reports_no_cache() {
        let event = br#"{"metadata":{"usage":{"inputTokens":7,"outputTokens":11}}}"#;

        assert_eq!(
            parse_bedrock_event(event),
            StreamingTokens {
                input: Some(7),
                output: Some(11),
                cache_read: None,
                cache_write: None,
                reasoning: None,
            }
        );
    }

    // -------------------------------------------------------------------------
    // Reasoning / thinking token breakdown
    // -------------------------------------------------------------------------

    #[test]
    fn anthropic_message_delta_reports_thinking_tokens() {
        let event = br#"{"type":"message_delta","usage":{
            "output_tokens":1500,
            "output_tokens_details":{"thinking_tokens":1200}
        }}"#;

        assert_eq!(
            parse_anthropic_event(event),
            StreamingTokens {
                input: None,
                output: Some(1500),
                cache_read: None,
                cache_write: None,
                reasoning: Some(1200),
            }
        );
    }

    #[test]
    fn anthropic_message_delta_thinking_zero_is_reported() {
        let event = br#"{"type":"message_delta","usage":{
            "output_tokens":42,
            "output_tokens_details":{"thinking_tokens":0}
        }}"#;

        assert_eq!(
            parse_anthropic_event(event),
            StreamingTokens {
                input: None,
                output: Some(42),
                cache_read: None,
                cache_write: None,
                reasoning: Some(0),
            }
        );
    }

    #[test]
    fn anthropic_message_start_reports_no_reasoning() {
        let event = br#"{"type":"message_start","message":{"usage":{"input_tokens":25}}}"#;

        assert_eq!(
            parse_anthropic_event(event).reasoning,
            None,
            "message_start never carries thinking tokens"
        );
    }
}
