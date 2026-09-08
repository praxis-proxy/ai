// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Provider-specific JSON parsing for the token usage filters.

use serde::Deserialize;

use super::TokenUsage;

/// Cache write counts are not reported by every provider.
///
/// `OpenAI` Chat Completions and Google expose how much of the prompt was *read*
/// from their cache but not how much was written to it, so those parsers leave
/// the cache write count absent rather than claiming a zero the provider never
/// reported.
const NO_CACHE_WRITE: Option<u64> = None;

// -----------------------------------------------------------------------------
// OpenAI / Azure
// -----------------------------------------------------------------------------

/// Fields needed from an `OpenAI` / Azure JSON response or SSE payload.
#[derive(Deserialize)]
struct OpenAiEnvelope {
    /// Top-level usage from Chat Completions or a non-streaming Responses API response.
    usage: Option<OpenAiTopLevelUsage>,

    /// Nested response from a Responses API streaming event.
    response: Option<ResponsesApiResponse>,
}

/// Top-level usage formats accepted by the `OpenAI` provider.
#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiTopLevelUsage {
    /// Chat Completions usage.
    ChatCompletions(ChatCompletionsUsage),
    /// Non-streaming Responses API usage.
    Responses(ResponsesApiUsage),
}

/// `OpenAI` Chat Completions usage object.
#[derive(Deserialize)]
struct ChatCompletionsUsage {
    /// Tokens in the prompt.
    prompt_tokens: u64,

    /// Tokens in the completion.
    completion_tokens: u64,

    /// Total tokens (optional, can be calculated).
    total_tokens: Option<u64>,

    /// Breakdown of the prompt tokens (prompt caching).
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,

    /// Breakdown of the completion tokens (reasoning models).
    completion_tokens_details: Option<OpenAiCompletionTokensDetails>,
}

/// `OpenAI` prompt token breakdown.
#[derive(Deserialize)]
struct OpenAiPromptTokensDetails {
    /// Tokens read from cache, already counted in `prompt_tokens`.
    cached_tokens: Option<u64>,
}

/// `OpenAI` completion token breakdown.
#[derive(Deserialize)]
struct OpenAiCompletionTokensDetails {
    /// Reasoning tokens, already counted in `completion_tokens`.
    reasoning_tokens: Option<u64>,
}

/// Parses `OpenAI`/Azure response format.
///
/// Supports both Chat Completions format (`usage.prompt_tokens`) and
/// Responses API format (`response.usage.input_tokens`). The Responses
/// API nests usage under a `response` wrapper and uses different field
/// names. Cached input is already counted in the input total for both formats,
/// so cache reads and writes are recorded as breakdowns rather than added to
/// it. Reasoning tokens are likewise a breakdown of `completion_tokens`
/// (Chat Completions) or `output_tokens` (Responses API), not an addition.
pub(super) fn parse_openai(body: &[u8]) -> Option<TokenUsage> {
    let envelope: OpenAiEnvelope = serde_json::from_slice(body).ok()?;

    if let Some(usage) = envelope.usage {
        return Some(match usage {
            OpenAiTopLevelUsage::ChatCompletions(usage) => chat_completions_usage(usage),
            OpenAiTopLevelUsage::Responses(usage) => responses_api_usage(usage),
        });
    }

    envelope.response?.usage.map(responses_api_usage)
}

/// Converts a Chat Completions usage object.
fn chat_completions_usage(usage: ChatCompletionsUsage) -> TokenUsage {
    let cache_read = usage.prompt_tokens_details.and_then(|details| details.cached_tokens);
    let reasoning = usage
        .completion_tokens_details
        .and_then(|details| details.reasoning_tokens);
    TokenUsage::new(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        .with_cache(cache_read, NO_CACHE_WRITE)
        .with_reasoning(reasoning)
}

/// Converts a Responses API usage object, keeping cache breakdowns out of the
/// totals.
fn responses_api_usage(usage: ResponsesApiUsage) -> TokenUsage {
    // `input_tokens` already includes cached reads and writes, so both counts
    // are recorded as breakdowns of the input rather than added to it.
    // Reasoning tokens are a breakdown of `output_tokens`, not an addition.
    let (cache_read, cache_write) = usage.input_tokens_details.map_or((None, None), |details| {
        (details.cached_tokens, details.cache_write_tokens)
    });
    let reasoning = usage.output_tokens_details.and_then(|details| details.reasoning_tokens);
    TokenUsage::new(usage.input_tokens, usage.output_tokens, usage.total_tokens)
        .with_cache(cache_read, cache_write)
        .with_reasoning(reasoning)
}

/// Inner response object in Responses API events.
#[derive(Deserialize)]
struct ResponsesApiResponse {
    /// Token usage statistics.
    usage: Option<ResponsesApiUsage>,
}

/// Responses API usage format (uses `input_tokens`/`output_tokens`).
#[derive(Deserialize)]
struct ResponsesApiUsage {
    /// Input tokens.
    input_tokens: u64,
    /// Output tokens.
    output_tokens: u64,
    /// Total tokens.
    total_tokens: Option<u64>,
    /// Breakdown of the input tokens (prompt caching).
    input_tokens_details: Option<ResponsesApiInputTokensDetails>,
    /// Breakdown of the output tokens (reasoning models).
    output_tokens_details: Option<ResponsesApiOutputTokensDetails>,
}

/// Responses API input token breakdown.
#[derive(Deserialize)]
struct ResponsesApiInputTokensDetails {
    /// Tokens read from cache, already counted in `input_tokens`.
    cached_tokens: Option<u64>,
    /// Tokens written to cache, already counted in `input_tokens`.
    cache_write_tokens: Option<u64>,
}

/// Responses API output token breakdown.
#[derive(Deserialize)]
struct ResponsesApiOutputTokensDetails {
    /// Reasoning tokens, already counted in `output_tokens`.
    reasoning_tokens: Option<u64>,
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

    /// Breakdown of the output tokens (extended thinking).
    output_tokens_details: Option<AnthropicOutputTokensDetails>,
}

/// Anthropic output token breakdown.
#[derive(Deserialize)]
struct AnthropicOutputTokensDetails {
    /// Thinking tokens, already counted in `output_tokens`.
    thinking_tokens: Option<u64>,
}

/// Parses `Anthropic` Claude response format.
///
/// When prompt caching is enabled, `input_tokens` only contains tokens after
/// the cache breakpoint. The actual total is the sum of all input token fields,
/// and the cache fields are also kept as a breakdown of that total. Thinking
/// tokens are a breakdown of `output_tokens`, not an addition.
pub(super) fn parse_anthropic(body: &[u8]) -> Option<TokenUsage> {
    let response: AnthropicResponse = serde_json::from_slice(body).ok()?;
    let usage = response.usage?;
    let cache_write = usage.cache_creation_input_tokens;
    let cache_read = usage.cache_read_input_tokens;
    let reasoning = usage.output_tokens_details.and_then(|details| details.thinking_tokens);
    let actual_input = usage
        .input_tokens
        .saturating_add(cache_write.unwrap_or(0))
        .saturating_add(cache_read.unwrap_or(0));
    Some(
        TokenUsage::new(actual_input, usage.output_tokens, None)
            .with_cache(cache_read, cache_write)
            .with_reasoning(reasoning),
    )
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

    /// Tokens in the candidates (output). Absent in safety-filtered responses.
    candidates_token_count: Option<u64>,

    /// Total tokens (optional, can be calculated).
    total_token_count: Option<u64>,

    /// Cached tokens, already counted in `prompt_token_count` (context caching).
    cached_content_token_count: Option<u64>,

    /// Thinking tokens. Unlike OpenAI/Anthropic, these are *not* included in
    /// `candidates_token_count`; they are billed separately and already in
    /// `total_token_count` when the provider sends that field.
    thoughts_token_count: Option<u64>,
}

/// Parses Google `Gemini` response format.
///
/// `promptTokenCount` already includes any cached tokens, so the cache read
/// count is recorded as a breakdown of the input rather than added to it.
/// Thinking tokens are reported separately from candidate output. When the
/// provider omits `totalTokenCount`, the fallback total includes thoughts
/// so billing and quota consumers do not undercount.
pub(super) fn parse_google(body: &[u8]) -> Option<TokenUsage> {
    let response: GoogleResponse = serde_json::from_slice(body).ok()?;
    let usage = response.usage_metadata?;
    let cache_read = usage.cached_content_token_count;
    let output = usage.candidates_token_count.unwrap_or(0);
    let thoughts = usage.thoughts_token_count.unwrap_or(0);
    let total = usage
        .total_token_count
        .unwrap_or_else(|| usage.prompt_token_count.saturating_add(output).saturating_add(thoughts));
    Some(
        TokenUsage::new(usage.prompt_token_count, output, Some(total))
            .with_cache(cache_read, NO_CACHE_WRITE)
            .with_reasoning(usage.thoughts_token_count),
    )
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
///
/// # Prompt Caching
///
/// The Converse API reports cache counts under a different shape than the one
/// parsed here, so no cache breakdown is recorded for it. It also has no
/// documented reasoning-token field. Claude via `InvokeModel` gets both the
/// cache and thinking breakdowns through the Anthropic fallback below.
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
    fn openai_responses_api_completed_event() {
        let json = br#"{"type":"response.completed","response":{"id":"resp_123","usage":{"input_tokens":150,"output_tokens":42,"total_tokens":192}}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(
            usage.input_tokens(),
            150,
            "should parse input_tokens from Responses API"
        );
        assert_eq!(usage.output_tokens(), 42);
        assert_eq!(usage.total_tokens(), 192);
    }

    #[test]
    fn openai_responses_api_without_total() {
        let json = br#"{"response":{"usage":{"input_tokens":10,"output_tokens":20}}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(usage.input_tokens(), 10);
        assert_eq!(usage.output_tokens(), 20);
        assert_eq!(usage.total_tokens(), 30, "should compute total when absent");
    }

    #[test]
    fn openai_responses_api_cached_tokens() {
        let json = br#"{"type":"response.completed","response":{"id":"resp_123","usage":{"input_tokens":150,"output_tokens":42,"total_tokens":192,"input_tokens_details":{"cached_tokens":120,"cache_write_tokens":30}}}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(usage.input_tokens(), 150, "cached input stays part of the input total");
        assert_eq!(usage.output_tokens(), 42);
        assert_eq!(usage.total_tokens(), 192);
        assert_eq!(
            usage.cache_read_tokens(),
            Some(120),
            "cached input should be reported as a cache-read breakdown"
        );
        assert_eq!(
            usage.cache_write_tokens(),
            Some(30),
            "cache writes should be reported as a breakdown of Responses API input"
        );
    }

    #[test]
    fn openai_responses_api_non_streaming() {
        let json = br#"{"id":"resp_123","object":"response","usage":{"input_tokens":150,"output_tokens":42,"total_tokens":192}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(
            usage.input_tokens(),
            150,
            "should parse non-streaming Responses API with top-level usage"
        );
        assert_eq!(usage.output_tokens(), 42);
        assert_eq!(usage.total_tokens(), 192);
    }

    #[test]
    fn openai_responses_api_non_streaming_without_total() {
        let json = br#"{"id":"resp_123","object":"response","usage":{"input_tokens":10,"output_tokens":20}}"#;
        let usage = parse_openai(json).unwrap();
        assert_eq!(usage.input_tokens(), 10);
        assert_eq!(usage.output_tokens(), 20);
        assert_eq!(
            usage.total_tokens(),
            30,
            "non-streaming Responses API should compute total when absent"
        );
    }

    #[test]
    fn openai_responses_api_null_response_returns_none() {
        let json = br#"{"type":"response.output_item.added","response":null}"#;
        assert!(parse_openai(json).is_none(), "null response should return None");
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
    fn google_missing_total_includes_thoughts() {
        let json = br#"{"usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 80,
            "thoughtsTokenCount": 200
        }}"#;
        let usage = parse_google(json).unwrap();

        assert_eq!(usage.output_tokens(), 80, "output stays candidatesTokenCount");
        assert_eq!(usage.reasoning_tokens(), Some(200));
        assert_eq!(
            usage.total_tokens(),
            330,
            "fallback total must include thoughts when totalTokenCount is absent"
        );
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

    #[test]
    fn google_missing_candidates_token_count_defaults_to_zero() {
        let json = br#"{"usageMetadata": {"promptTokenCount": 42, "totalTokenCount": 42}}"#;
        let usage = parse_google(json).unwrap();
        assert_eq!(usage.input_tokens(), 42, "input tokens should be preserved");
        assert_eq!(
            usage.output_tokens(),
            0,
            "absent candidatesTokenCount should default to 0"
        );
        assert_eq!(usage.total_tokens(), 42);
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

    // -------------------------------------------------------------------------
    // Prompt cache breakdown
    // -------------------------------------------------------------------------

    #[test]
    fn openai_cached_tokens_are_a_subset_of_prompt_tokens() {
        let json = br#"{"usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900}
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(usage.input_tokens(), 1000, "cached tokens are already in prompt_tokens");
        assert_eq!(usage.total_tokens(), 1050, "total should be prompt + completion");
        assert_eq!(usage.cache_read_tokens(), Some(900), "cached_tokens is the cache read");
        assert_eq!(
            usage.cache_write_tokens(),
            None,
            "OpenAI has no cache write field, so the count is absent rather than zero"
        );
    }

    #[test]
    fn openai_cached_tokens_zero_is_reported_not_absent() {
        let json = br#"{"usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 0}
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            Some(0),
            "an explicit zero is a reported cache miss, distinct from an absent count"
        );
    }

    #[test]
    fn openai_without_prompt_tokens_details_reports_no_cache() {
        let json = br#"{"usage": {"prompt_tokens": 10, "completion_tokens": 20}}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            None,
            "no prompt_tokens_details means no cache information"
        );
        assert_eq!(usage.cache_write_tokens(), None, "OpenAI never reports cache writes");
    }

    #[test]
    fn openai_null_prompt_tokens_details_reports_no_cache() {
        let json = br#"{"usage": {"prompt_tokens": 10, "completion_tokens": 20, "prompt_tokens_details": null}}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            None,
            "null prompt_tokens_details means no cache information"
        );
    }

    #[test]
    fn anthropic_reports_both_cache_directions() {
        let json = br#"{"usage": {
            "input_tokens": 50,
            "output_tokens": 100,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 3000
        }}"#;
        let usage = parse_anthropic(json).unwrap();

        assert_eq!(usage.input_tokens(), 3250, "input is the sum of all input fields");
        assert_eq!(
            usage.cache_read_tokens(),
            Some(3000),
            "cache_read_input_tokens is the cache read"
        );
        assert_eq!(
            usage.cache_write_tokens(),
            Some(200),
            "cache_creation_input_tokens is the cache write"
        );
    }

    #[test]
    fn anthropic_without_cache_fields_reports_no_cache() {
        let json = br#"{"usage": {"input_tokens": 20, "output_tokens": 30}}"#;
        let usage = parse_anthropic(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            None,
            "an omitted cache read field means no cache information"
        );
        assert_eq!(
            usage.cache_write_tokens(),
            None,
            "an omitted cache write field means no cache information"
        );
    }

    #[test]
    fn google_cached_content_tokens_are_a_subset_of_prompt_tokens() {
        let json = br#"{"usageMetadata": {
            "promptTokenCount": 1200,
            "candidatesTokenCount": 40,
            "cachedContentTokenCount": 1100
        }}"#;
        let usage = parse_google(json).unwrap();

        assert_eq!(
            usage.input_tokens(),
            1200,
            "cached tokens are already in promptTokenCount"
        );
        assert_eq!(
            usage.cache_read_tokens(),
            Some(1100),
            "cachedContentTokenCount is the cache read"
        );
        assert_eq!(
            usage.cache_write_tokens(),
            None,
            "Google has no cache write field, so the count is absent rather than zero"
        );
    }

    #[test]
    fn google_without_cached_content_reports_no_cache() {
        let json = br#"{"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}}"#;
        let usage = parse_google(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            None,
            "an omitted cachedContentTokenCount means no cache information"
        );
    }

    #[test]
    fn bedrock_converse_reports_no_cache() {
        let json = br#"{"usage": {"inputTokens": 10, "outputTokens": 20}}"#;
        let usage = parse_bedrock(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            None,
            "the Converse cache shape is not parsed, so no cache read is claimed"
        );
        assert_eq!(
            usage.cache_write_tokens(),
            None,
            "the Converse cache shape is not parsed, so no cache write is claimed"
        );
        assert_eq!(
            usage.reasoning_tokens(),
            None,
            "the Converse usage shape has no reasoning field"
        );
    }

    #[test]
    fn bedrock_anthropic_fallback_reports_cache_breakdown() {
        let json = br#"{"usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 200
        }}"#;
        let usage = parse_bedrock(json).unwrap();

        assert_eq!(
            usage.cache_read_tokens(),
            Some(200),
            "the Anthropic fallback carries the cache read through"
        );
        assert_eq!(
            usage.cache_write_tokens(),
            Some(100),
            "the Anthropic fallback carries the cache write through"
        );
    }

    // -------------------------------------------------------------------------
    // Reasoning / thinking token breakdown
    // -------------------------------------------------------------------------

    #[test]
    fn openai_reasoning_tokens_are_a_subset_of_completion_tokens() {
        let json = br#"{"usage": {
            "prompt_tokens": 120,
            "completion_tokens": 800,
            "total_tokens": 920,
            "completion_tokens_details": {"reasoning_tokens": 640}
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(usage.input_tokens(), 120);
        assert_eq!(
            usage.output_tokens(),
            800,
            "reasoning is already in completion_tokens; output must not change"
        );
        assert_eq!(usage.total_tokens(), 920);
        assert_eq!(usage.reasoning_tokens(), Some(640));
    }

    #[test]
    fn openai_reasoning_tokens_zero_is_reported_not_absent() {
        let json = br#"{"usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 0}
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.reasoning_tokens(),
            Some(0),
            "an explicit zero is a reported non-reasoning completion, distinct from absent"
        );
    }

    #[test]
    fn openai_without_completion_tokens_details_reports_no_reasoning() {
        let json = br#"{"usage": {"prompt_tokens": 10, "completion_tokens": 20}}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.reasoning_tokens(),
            None,
            "no completion_tokens_details means no reasoning information"
        );
    }

    #[test]
    fn openai_null_completion_tokens_details_reports_no_reasoning() {
        let json = br#"{"usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": null
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(usage.reasoning_tokens(), None);
    }

    #[test]
    fn openai_reports_cache_and_reasoning_together() {
        let json = br#"{"usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 800,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens_details": {"reasoning_tokens": 640}
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(usage.cache_read_tokens(), Some(900));
        assert_eq!(usage.reasoning_tokens(), Some(640));
        assert_eq!(usage.output_tokens(), 800);
        assert_eq!(usage.input_tokens(), 1000);
    }

    #[test]
    fn openai_responses_api_reasoning_tokens_are_a_subset_of_output_tokens() {
        let json = br#"{"type":"response.completed","response":{"usage":{
            "input_tokens":120,
            "output_tokens":800,
            "total_tokens":920,
            "output_tokens_details":{"reasoning_tokens":640}
        }}}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(usage.input_tokens(), 120);
        assert_eq!(
            usage.output_tokens(),
            800,
            "reasoning is already in output_tokens; output must not change"
        );
        assert_eq!(usage.total_tokens(), 920);
        assert_eq!(usage.reasoning_tokens(), Some(640));
    }

    #[test]
    fn openai_responses_api_non_streaming_reasoning_tokens() {
        let json = br#"{"id":"resp_123","object":"response","usage":{
            "input_tokens":120,
            "output_tokens":800,
            "output_tokens_details":{"reasoning_tokens":640}
        }}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(usage.reasoning_tokens(), Some(640));
        assert_eq!(usage.output_tokens(), 800);
    }

    #[test]
    fn openai_responses_api_reasoning_tokens_zero_is_reported_not_absent() {
        let json = br#"{"response":{"usage":{
            "input_tokens":10,
            "output_tokens":20,
            "output_tokens_details":{"reasoning_tokens":0}
        }}}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.reasoning_tokens(),
            Some(0),
            "an explicit zero is a reported non-reasoning completion, distinct from absent"
        );
    }

    #[test]
    fn openai_responses_api_without_output_tokens_details_reports_no_reasoning() {
        let json = br#"{"response":{"usage":{"input_tokens":10,"output_tokens":20}}}"#;
        let usage = parse_openai(json).unwrap();

        assert_eq!(
            usage.reasoning_tokens(),
            None,
            "no output_tokens_details means no reasoning information"
        );
    }

    #[test]
    fn anthropic_thinking_tokens_are_a_subset_of_output_tokens() {
        let json = br#"{"usage": {
            "input_tokens": 200,
            "output_tokens": 1500,
            "output_tokens_details": {"thinking_tokens": 1200}
        }}"#;
        let usage = parse_anthropic(json).unwrap();

        assert_eq!(usage.input_tokens(), 200);
        assert_eq!(
            usage.output_tokens(),
            1500,
            "thinking is already in output_tokens; output must not change"
        );
        assert_eq!(usage.reasoning_tokens(), Some(1200));
    }

    #[test]
    fn anthropic_thinking_tokens_zero_is_reported_not_absent() {
        let json = br#"{"usage": {
            "input_tokens": 20,
            "output_tokens": 30,
            "output_tokens_details": {"thinking_tokens": 0}
        }}"#;
        let usage = parse_anthropic(json).unwrap();

        assert_eq!(usage.reasoning_tokens(), Some(0));
    }

    #[test]
    fn anthropic_without_output_tokens_details_reports_no_reasoning() {
        let json = br#"{"usage": {"input_tokens": 20, "output_tokens": 30}}"#;
        let usage = parse_anthropic(json).unwrap();

        assert_eq!(usage.reasoning_tokens(), None);
    }

    #[test]
    fn google_thoughts_tokens_are_reported_separately_from_candidates() {
        let json = br#"{"usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 80,
            "thoughtsTokenCount": 200,
            "totalTokenCount": 330
        }}"#;
        let usage = parse_google(json).unwrap();

        assert_eq!(usage.input_tokens(), 50);
        assert_eq!(
            usage.output_tokens(),
            80,
            "candidatesTokenCount stays the output; thoughts are not folded in"
        );
        assert_eq!(usage.total_tokens(), 330, "provider total includes thoughts");
        assert_eq!(usage.reasoning_tokens(), Some(200));
    }

    #[test]
    fn google_without_thoughts_reports_no_reasoning() {
        let json = br#"{"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}}"#;
        let usage = parse_google(json).unwrap();

        assert_eq!(usage.reasoning_tokens(), None);
    }

    #[test]
    fn bedrock_anthropic_fallback_reports_thinking_tokens() {
        let json = br#"{"usage": {
            "input_tokens": 10,
            "output_tokens": 1500,
            "output_tokens_details": {"thinking_tokens": 1200}
        }}"#;
        let usage = parse_bedrock(json).unwrap();

        assert_eq!(usage.output_tokens(), 1500);
        assert_eq!(
            usage.reasoning_tokens(),
            Some(1200),
            "Claude via InvokeModel inherits Anthropic thinking extraction"
        );
    }
}
