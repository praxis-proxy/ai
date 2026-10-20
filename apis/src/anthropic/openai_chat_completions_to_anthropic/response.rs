// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages response to OpenAI Chat Completions JSON conversion.

use serde_json::{Map, Value, json};

/// Map Anthropic terminal reasons to the OpenAI Chat Completions finish reason.
pub(crate) fn map_stop_reason(stop_reason: &str) -> Result<&'static str, String> {
    match stop_reason {
        "end_turn" | "stop_sequence" => Ok("stop"),
        "tool_use" => Ok("tool_calls"),
        "max_tokens" | "model_context_window_exceeded" => Ok("length"),
        "refusal" => Ok("content_filter"),
        other => Err(format!("unsupported Anthropic stop_reason {other:?}")),
    }
}

/// Translate one successful Anthropic Messages response.
#[expect(
    clippy::too_many_lines,
    reason = "The response conversion validates the complete envelope before serialization"
)]
pub(crate) fn translate_response(bytes: &[u8], client_model: &str) -> Result<Vec<u8>, String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid Anthropic response JSON: {error}"))?;
    let object = value.as_object().ok_or("Anthropic response must be a JSON object")?;
    if object.get("type").and_then(Value::as_str) != Some("message")
        || object.get("role").and_then(Value::as_str) != Some("assistant")
    {
        return Err("Anthropic response is not an assistant message".to_owned());
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("Anthropic response requires id")?;
    let (text, tool_calls) = translate_content_blocks(
        object
            .get("content")
            .and_then(Value::as_array)
            .ok_or("Anthropic response requires content array")?,
    )?;
    let stop_reason = object
        .get("stop_reason")
        .and_then(Value::as_str)
        .ok_or("Anthropic response requires stop_reason")?;
    let finish_reason = map_stop_reason(stop_reason)?;
    let usage = object
        .get("usage")
        .and_then(Value::as_object)
        .ok_or("Anthropic response requires usage object")?;
    let usage = translate_usage(usage)?;
    let content = if text.is_empty() && !tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(text)
    };
    let mut message = Map::from_iter([
        ("role".to_owned(), Value::String("assistant".to_owned())),
        ("content".to_owned(), content),
    ]);
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    let mut usage_json = Map::new();
    usage_json.insert("prompt_tokens".to_owned(), json!(usage.prompt_tokens));
    usage_json.insert("completion_tokens".to_owned(), json!(usage.completion_tokens));
    usage_json.insert("total_tokens".to_owned(), json!(usage.total_tokens));
    if let Some(cached) = usage.cached_tokens {
        usage_json.insert("prompt_tokens_details".to_owned(), json!({"cached_tokens":cached}));
    }
    let result = json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion",
        "created": 0,
        "model": client_model,
        "choices": [{"index":0,"message":Value::Object(message),"finish_reason":finish_reason}],
        "usage": Value::Object(usage_json)
    });
    serde_json::to_vec(&result).map_err(|error| format!("serialize Chat Completions response: {error}"))
}

/// Translate text and function blocks while rejecting unsupported modalities.
fn translate_content_blocks(content: &[Value]) -> Result<(String, Vec<Value>), String> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in content {
        let block = block
            .as_object()
            .ok_or("Anthropic response content block must be an object")?;
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or("Anthropic text block requires text")?,
            ),
            Some("tool_use") => tool_calls.push(translate_tool_use(block)?),
            Some(kind) => return Err(format!("unsupported Anthropic response content block {kind:?}")),
            None => return Err("Anthropic response content block requires type".to_owned()),
        }
    }
    Ok((text, tool_calls))
}

/// Preserve the token counts representable by Chat Completions.
/// Token counts and cache metadata returned to a Chat Completions client.
struct UsageCounts {
    /// Input tokens reported by Anthropic.
    prompt_tokens: u64,
    /// Output tokens reported by Anthropic.
    completion_tokens: u64,
    /// Sum of input and output tokens.
    total_tokens: u64,
    /// Optional cache-read count when the provider reported it.
    cached_tokens: Option<u64>,
}

/// Translate the usage counters representable by Chat Completions.
fn translate_usage(usage: &Map<String, Value>) -> Result<UsageCounts, String> {
    let prompt_tokens = total_input_tokens(usage)?;
    let completion_tokens = positive_or_zero(usage, "output_tokens")?;
    let total_tokens = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or("Anthropic usage token count overflow")?;
    let cached_tokens = usage
        .get("cache_read_input_tokens")
        .filter(|value| !value.is_null())
        .map(|_| optional_usage(usage, "cache_read_input_tokens"))
        .transpose()?;
    Ok(UsageCounts {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_tokens,
    })
}

/// Anthropic reports uncached, cache-read, and cache-created input separately.
pub(crate) fn total_input_tokens(usage: &Map<String, Value>) -> Result<u64, String> {
    let input = positive_or_zero(usage, "input_tokens")?;
    let read = optional_usage(usage, "cache_read_input_tokens")?;
    let created = optional_usage(usage, "cache_creation_input_tokens")?;
    input
        .checked_add(read)
        .and_then(|sum| sum.checked_add(created))
        .ok_or_else(|| "Anthropic usage token count overflow".to_owned())
}

/// Read an optional non-negative Anthropic usage counter, defaulting absence to zero.
fn optional_usage(usage: &Map<String, Value>, key: &str) -> Result<u64, String> {
    usage.get(key).filter(|value| !value.is_null()).map_or(Ok(0), |value| {
        value
            .as_u64()
            .ok_or_else(|| format!("Anthropic usage requires non-negative integer {key}"))
    })
}

/// Convert a provider error body into the stable OpenAI error envelope.
pub(crate) fn translate_error(bytes: &[u8], status: u16) -> Vec<u8> {
    let parsed: Result<Value, _> = serde_json::from_slice(bytes);
    let message = parsed
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "Anthropic provider returned an error".to_owned());
    let error_type = if status == 429 {
        "rate_limit_error"
    } else if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    serde_json::to_vec(&json!({"error":{"message":message,"type":error_type,"code":status}})).unwrap_or_else(|_| {
        br#"{"error":{"message":"Anthropic provider returned an error","type":"server_error"}}"#.to_vec()
    })
}

/// Convert one Anthropic `tool_use` block to an OpenAI function call.
fn translate_tool_use(block: &Map<String, Value>) -> Result<Value, String> {
    let id = block
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("Anthropic tool_use requires id")?;
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("Anthropic tool_use requires name")?;
    let input = block
        .get("input")
        .and_then(Value::as_object)
        .ok_or("Anthropic tool_use input must be a JSON object")?;
    let arguments = serde_json::to_string(input).map_err(|error| format!("serialize tool arguments: {error}"))?;
    Ok(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}}))
}

/// Read a non-negative integer usage member.
fn positive_or_zero(object: &Map<String, Value>, key: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("Anthropic usage requires non-negative integer {key}"))
}
