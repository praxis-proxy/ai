// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Provider-specific JSON parsing for the token usage filters.

use serde::Deserialize;

use super::TokenUsage;

// -----------------------------------------------------------------------------
// OpenAI / Azure
// -----------------------------------------------------------------------------

/// `OpenAI` / Azure `OpenAI` response format.
#[derive(Deserialize)]
struct OpenAiResponse {
    /// Token usage statistics.
    usage: Option<OpenAiUsage>,
}

/// `OpenAI` usage object.
#[derive(Deserialize)]
struct OpenAiUsage {
    /// Tokens in the prompt.
    prompt_tokens: u64,

    /// Tokens in the completion.
    completion_tokens: u64,

    /// Total tokens (optional, can be calculated).
    total_tokens: Option<u64>,
}

/// Parses `OpenAI`/Azure response format.
pub(super) fn parse_openai(body: &[u8]) -> Option<TokenUsage> {
    let response: OpenAiResponse = serde_json::from_slice(body).ok()?;
    let usage = response.usage?;
    Some(TokenUsage::new(
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    ))
}

// -----------------------------------------------------------------------------
// Anthropic
// -----------------------------------------------------------------------------

/// `Anthropic` Claude response format.
#[derive(Deserialize)]
struct AnthropicResponse {
    /// Token usage statistics.
    usage: Option<AnthropicUsage>,
}

/// `Anthropic` usage object.
#[derive(Deserialize)]
struct AnthropicUsage {
    /// Tokens in the input (excludes cached tokens when caching is active).
    input_tokens: u64,

    /// Tokens in the output.
    output_tokens: u64,

    /// Tokens written to cache (prompt caching).
    cache_creation_input_tokens: Option<u64>,

    /// Tokens read from cache (prompt caching).
    cache_read_input_tokens: Option<u64>,
}

/// Parses `Anthropic` Claude response format.
///
/// When prompt caching is enabled, `input_tokens` only contains tokens after
/// the cache breakpoint. The actual total is the sum of all input token fields.
pub(super) fn parse_anthropic(body: &[u8]) -> Option<TokenUsage> {
    let response: AnthropicResponse = serde_json::from_slice(body).ok()?;
    let usage = response.usage?;
    let actual_input = usage
        .input_tokens
        .saturating_add(usage.cache_creation_input_tokens.unwrap_or(0))
        .saturating_add(usage.cache_read_input_tokens.unwrap_or(0));
    Some(TokenUsage::new(actual_input, usage.output_tokens, None))
}

// -----------------------------------------------------------------------------
// Google Gemini
// -----------------------------------------------------------------------------

/// Google `Gemini` response format.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleResponse {
    /// Token usage metadata.
    usage_metadata: Option<GoogleUsageMetadata>,
}

/// Google `Gemini` usage metadata object.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleUsageMetadata {
    /// Tokens in the prompt.
    prompt_token_count: u64,

    /// Tokens in the candidates (output).
    candidates_token_count: u64,

    /// Total tokens (optional, can be calculated).
    total_token_count: Option<u64>,
}

/// Parses Google `Gemini` response format.
pub(super) fn parse_google(body: &[u8]) -> Option<TokenUsage> {
    let response: GoogleResponse = serde_json::from_slice(body).ok()?;
    let usage = response.usage_metadata?;
    Some(TokenUsage::new(
        usage.prompt_token_count,
        usage.candidates_token_count,
        usage.total_token_count,
    ))
}

// -----------------------------------------------------------------------------
// AWS Bedrock
// -----------------------------------------------------------------------------

/// AWS `Bedrock` Converse API response format (fields in `usage` object).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockConverseResponse {
    /// Token usage statistics.
    usage: Option<BedrockConverseUsage>,
}

/// `Bedrock` Converse API usage object.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockConverseUsage {
    /// Tokens in the input.
    input_tokens: u64,

    /// Tokens in the output.
    output_tokens: u64,

    /// Total tokens (optional).
    total_tokens: Option<u64>,
}

/// Parses AWS `Bedrock` response format.
///
/// # Supported Formats
///
/// 1. **Converse API** (recommended): `usage.inputTokens`, `usage.outputTokens`
///    - AWS's unified API that works with all Bedrock models
///    - Always returns a consistent format regardless of underlying model
///
/// 2. **Claude via `InvokeModel`**: `usage.input_tokens`, `usage.output_tokens`
///    - Claude models via `InvokeModel` use the same format as direct Anthropic API
///
/// # Not Supported
///
/// Other models via `InvokeModel` have different response formats:
/// - Titan: `inputTextTokenCount`, `results[0].tokenCount`
/// - Llama: `prompt_token_count`, `generation_token_count`
/// - Cohere: token counts in HTTP headers
///
/// For these models, use the Converse API or submit a follow-up issue to add support.
pub(super) fn parse_bedrock(body: &[u8]) -> Option<TokenUsage> {
    // Try Converse API format first (AWS recommended, works with all models)
    if let Ok(response) = serde_json::from_slice::<BedrockConverseResponse>(body)
        && let Some(usage) = response.usage
    {
        return Some(TokenUsage::new(
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        ));
    }

    // Fall back to Claude/Anthropic format (Claude via InvokeModel)
    // Claude via Bedrock InvokeModel uses the same format as direct Anthropic API
    parse_anthropic(body)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // parse_openai edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn openai_missing_total_tokens_computes_sum() {
        let json = br#"{"usage": {"prompt_tokens": 10, "completion_tokens": 20}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(usage.input_tokens(), 10);
        assert_eq!(usage.output_tokens(), 20);
        assert_eq!(usage.total_tokens(), 30, "total should be computed as input + output");
    }

    #[test]
    fn openai_null_usage_field_returns_none() {
        let json = br#"{"usage": null}"#;
        assert!(parse_openai(json).is_none(), "null usage should return None");
    }

    #[test]
    fn openai_missing_usage_field_returns_none() {
        let json = br#"{"id": "chatcmpl-abc"}"#;
        assert!(parse_openai(json).is_none(), "absent usage should return None");
    }

    #[test]
    fn openai_usage_missing_required_field_returns_none() {
        // prompt_tokens present but completion_tokens absent
        let json = br#"{"usage": {"prompt_tokens": 10}}"#;
        assert!(
            parse_openai(json).is_none(),
            "usage with missing required field should return None"
        );
    }

    #[test]
    fn openai_zero_values() {
        let json = br#"{"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(usage.input_tokens(), 0);
        assert_eq!(usage.output_tokens(), 0);
        assert_eq!(usage.total_tokens(), 0);
    }

    // -------------------------------------------------------------------------
    // parse_anthropic edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn anthropic_only_cache_creation_tokens() {
        let json = br#"{"usage": {"input_tokens": 50, "output_tokens": 100, "cache_creation_input_tokens": 1000}}"#;
        let usage = parse_anthropic(json).unwrap();
        assert_eq!(usage.input_tokens(), 1050, "input should be 50 + 1000");
        assert_eq!(usage.output_tokens(), 100);
        assert_eq!(usage.total_tokens(), 1150);
    }

    #[test]
    fn anthropic_only_cache_read_tokens() {
        let json = br#"{"usage": {"input_tokens": 50, "output_tokens": 100, "cache_read_input_tokens": 3000}}"#;
        let usage = parse_anthropic(json).unwrap();
        assert_eq!(usage.input_tokens(), 3050, "input should be 50 + 3000");
        assert_eq!(usage.output_tokens(), 100);
        assert_eq!(usage.total_tokens(), 3150);
    }

    #[test]
    fn anthropic_both_cache_fields_zero() {
        let json = br#"{"usage": {"input_tokens": 50, "output_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}"#;
        let usage = parse_anthropic(json).unwrap();
        assert_eq!(usage.input_tokens(), 50, "zero cache tokens should not change input");
        assert_eq!(usage.output_tokens(), 100);
    }

    #[test]
    fn anthropic_no_cache_fields() {
        let json = br#"{"usage": {"input_tokens": 20, "output_tokens": 30}}"#;
        let usage = parse_anthropic(json).unwrap();
        assert_eq!(usage.input_tokens(), 20, "absent cache fields default to 0");
        assert_eq!(usage.output_tokens(), 30);
        assert_eq!(usage.total_tokens(), 50, "total computed as 20 + 30");
    }

    #[test]
    fn anthropic_saturating_add_prevents_overflow() {
        let json = format!(
            r#"{{"usage": {{"input_tokens": {max}, "output_tokens": 1, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 1}}}}"#,
            max = u64::MAX
        );
        let usage = parse_anthropic(json.as_bytes()).unwrap();
        assert_eq!(usage.input_tokens(), u64::MAX, "saturating_add should cap at u64::MAX");
    }

    #[test]
    fn anthropic_null_usage_returns_none() {
        let json = br#"{"usage": null}"#;
        assert!(parse_anthropic(json).is_none());
    }

    #[test]
    fn anthropic_missing_usage_returns_none() {
        let json = br#"{"type": "message"}"#;
        assert!(parse_anthropic(json).is_none());
    }

    // -------------------------------------------------------------------------
    // parse_google edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn google_missing_total_computes_sum() {
        let json = br#"{"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}}"#;
        let usage = parse_google(json).unwrap();
        assert_eq!(usage.total_tokens(), 30, "missing totalTokenCount should be computed");
    }

    #[test]
    fn google_null_usage_metadata_returns_none() {
        let json = br#"{"usageMetadata": null}"#;
        assert!(parse_google(json).is_none());
    }

    #[test]
    fn google_missing_usage_metadata_returns_none() {
        let json = br#"{"candidates": [{"content": {}}]}"#;
        assert!(parse_google(json).is_none());
    }

    #[test]
    fn google_zero_values() {
        let json = br#"{"usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}}"#;
        let usage = parse_google(json).unwrap();
        assert_eq!(usage.input_tokens(), 0);
        assert_eq!(usage.output_tokens(), 0);
        assert_eq!(usage.total_tokens(), 0);
    }

    // -------------------------------------------------------------------------
    // parse_bedrock edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn bedrock_converse_without_total_tokens() {
        let json = br#"{"usage": {"inputTokens": 10, "outputTokens": 20}}"#;
        let usage = parse_bedrock(json).unwrap();
        assert_eq!(usage.input_tokens(), 10);
        assert_eq!(usage.output_tokens(), 20);
        assert_eq!(usage.total_tokens(), 30, "missing totalTokens should be computed");
    }

    #[test]
    fn bedrock_converse_with_zero_total() {
        let json = br#"{"usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 0}}"#;
        let usage = parse_bedrock(json).unwrap();
        assert_eq!(usage.total_tokens(), 0, "explicit zero totalTokens should be preserved");
    }

    #[test]
    fn bedrock_falls_back_to_anthropic_format() {
        // snake_case fields => Converse parse sees missing fields, falls back to Anthropic
        let json = br#"{"usage": {"input_tokens": 25, "output_tokens": 75}}"#;
        let usage = parse_bedrock(json).unwrap();
        assert_eq!(usage.input_tokens(), 25);
        assert_eq!(usage.output_tokens(), 75);
        assert_eq!(usage.total_tokens(), 100);
    }

    #[test]
    fn bedrock_anthropic_fallback_with_cache_tokens() {
        let json = br#"{"usage": {"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 200}}"#;
        let usage = parse_bedrock(json).unwrap();
        assert_eq!(
            usage.input_tokens(),
            310,
            "Anthropic fallback should sum cache tokens: 10 + 100 + 200"
        );
    }

    #[test]
    fn bedrock_no_usage_at_all_returns_none() {
        let json = br#"{"output": {"message": {"role": "assistant"}}}"#;
        assert!(
            parse_bedrock(json).is_none(),
            "no usage in any format should return None"
        );
    }

    // -------------------------------------------------------------------------
    // Cross-provider: malformed/degenerate inputs
    // -------------------------------------------------------------------------

    #[test]
    fn all_parsers_empty_object_returns_none() {
        let json = b"{}";
        assert!(parse_openai(json).is_none());
        assert!(parse_anthropic(json).is_none());
        assert!(parse_google(json).is_none());
        assert!(parse_bedrock(json).is_none());
    }

    #[test]
    fn all_parsers_malformed_json_returns_none() {
        let json = b"{invalid json";
        assert!(parse_openai(json).is_none());
        assert!(parse_anthropic(json).is_none());
        assert!(parse_google(json).is_none());
        assert!(parse_bedrock(json).is_none());
    }

    #[test]
    fn all_parsers_null_body_returns_none() {
        let json = b"null";
        assert!(parse_openai(json).is_none());
        assert!(parse_anthropic(json).is_none());
        assert!(parse_google(json).is_none());
        assert!(parse_bedrock(json).is_none());
    }

    #[test]
    fn all_parsers_empty_body_returns_none() {
        let json = b"";
        assert!(parse_openai(json).is_none());
        assert!(parse_anthropic(json).is_none());
        assert!(parse_google(json).is_none());
        assert!(parse_bedrock(json).is_none());
    }

    #[test]
    fn all_parsers_usage_wrong_type_returns_none() {
        // usage/usageMetadata is a string instead of an object
        assert!(parse_openai(br#"{"usage": "not an object"}"#).is_none());
        assert!(parse_anthropic(br#"{"usage": "not an object"}"#).is_none());
        assert!(parse_google(br#"{"usageMetadata": "not an object"}"#).is_none());
        assert!(parse_bedrock(br#"{"usage": "not an object"}"#).is_none());
    }
}
