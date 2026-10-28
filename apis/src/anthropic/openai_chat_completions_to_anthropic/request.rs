// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Validated Chat Completions request conversion for Anthropic Messages.

use std::io::Read as _;

use base64::{engine::general_purpose::STANDARD, read::DecoderReader};
use serde_json::{Map, Value, json};
use url::Url;

/// Request details retained for response JSON and SSE conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RequestProfile {
    /// Client-visible model name from the original request.
    pub(crate) client_model: String,
    /// Whether the client requested a streaming response.
    pub(crate) stream: bool,
    /// Whether an OpenAI usage-only terminal chunk was requested.
    pub(crate) include_usage: bool,
}

/// Read and validate the client model in an OpenAI Chat Completions body.
pub(crate) fn client_model(bytes: &[u8]) -> Result<String, String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid Chat Completions JSON: {error}"))?;
    value
        .as_object()
        .ok_or("Chat Completions request must be a JSON object")?
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "Chat Completions request requires a non-empty string model".to_owned())
}

/// Rewrite only the model alias for an OpenAI-compatible selected provider.
pub(crate) fn rewrite_openai_target_model(bytes: &[u8], target_model: &str) -> Result<Vec<u8>, String> {
    let mut value: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid Chat Completions JSON: {error}"))?;
    let object = value
        .as_object_mut()
        .ok_or("Chat Completions request must be a JSON object")?;
    object
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .ok_or("Chat Completions request requires a non-empty string model")?;
    if target_model.trim().is_empty() {
        return Err("selected OpenAI-compatible provider target model is empty".to_owned());
    }
    object.insert("model".to_owned(), Value::String(target_model.to_owned()));
    serde_json::to_vec(&value).map_err(|error| format!("serialize Chat Completions request: {error}"))
}

/// Convert one OpenAI Chat Completions request to Anthropic Messages.
///
/// This deliberately supports a narrow, fidelity-preserving subset. Unknown
/// fields are rejected so provider-specific options cannot disappear silently.
#[expect(
    clippy::too_many_lines,
    reason = "The strict request mapping keeps validation adjacent to each field conversion"
)]
pub(crate) fn translate_request(bytes: &[u8], target_model: &str) -> Result<(Vec<u8>, RequestProfile), String> {
    let mut value: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid Chat Completions JSON: {error}"))?;
    let object = value
        .as_object_mut()
        .ok_or("Chat Completions request must be a JSON object")?;
    reject_unknown_fields(object)?;
    validate_ignored_legacy_fields(object)?;

    let client_model = take_string(object, "model")?;
    if client_model.trim().is_empty() {
        return Err("Chat Completions model must not be empty".to_owned());
    }
    let max_tokens = resolve_max_tokens(object)?;
    let client_messages = object
        .remove("messages")
        .ok_or("Chat Completions request requires messages")?;
    let (messages, system) = translate_messages(&client_messages)?;
    let mut output = Map::new();
    output.insert("model".to_owned(), Value::String(target_model.to_owned()));
    output.insert("max_tokens".to_owned(), Value::Number(max_tokens.into()));
    output.insert("messages".to_owned(), Value::Array(messages));
    if let Some(system) = system {
        output.insert("system".to_owned(), Value::String(system));
    }

    if let Some(stream) = object.remove("stream") {
        if !stream.is_boolean() {
            return Err("Chat Completions stream must be a boolean".to_owned());
        }
        output.insert("stream".to_owned(), stream);
    }
    for (source, destination) in [("temperature", "temperature"), ("top_p", "top_p")] {
        if let Some(value) = object.remove(source) {
            if !value.is_number() {
                return Err(format!("Chat Completions {source} must be numeric"));
            }
            output.insert(destination.to_owned(), value);
        }
    }
    if let Some(stop) = object.remove("stop") {
        output.insert("stop_sequences".to_owned(), normalize_stop(stop)?);
    }
    let tools = object.remove("tools");
    let tool_choice = object.remove("tool_choice");
    match tools {
        Some(Value::Array(tools)) => {
            let translated = translate_tools(&tools)?;
            if !translated.is_empty() {
                output.insert("tools".to_owned(), Value::Array(translated));
                if let Some(choice) = tool_choice.as_ref()
                    && let Some(choice) = translate_tool_choice(choice)?
                {
                    output.insert("tool_choice".to_owned(), choice);
                }
            } else if tool_choice.as_ref().is_some_and(|choice| !is_tool_choice_none(choice)) {
                return Err("tool_choice cannot be set when tools is empty".to_owned());
            }
        },
        Some(_) => return Err("Chat Completions tools must be an array".to_owned()),
        None if tool_choice.is_some() => return Err("tool_choice requires tools".to_owned()),
        None => {},
    }

    let profile = RequestProfile {
        client_model,
        stream: output.get("stream").and_then(Value::as_bool).unwrap_or(false),
        include_usage: include_usage_from(object)?,
    };
    let translated =
        serde_json::to_vec(&Value::Object(output)).map_err(|error| format!("serialize Anthropic request: {error}"))?;
    Ok((translated, profile))
}

/// Reject request members that cannot be preserved by the target API.
fn reject_unknown_fields(object: &Map<String, Value>) -> Result<(), String> {
    const ALLOWED: &[&str] = &[
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "n",
        "seed",
    ];
    if let Some(field) = object.keys().find(|key| !ALLOWED.contains(&key.as_str())) {
        return Err(format!(
            "unsupported Chat Completions field {field:?} cannot be preserved by Anthropic Messages"
        ));
    }
    Ok(())
}

/// Select the token limit using the compatibility precedence and default.
fn resolve_max_tokens(object: &mut Map<String, Value>) -> Result<u64, String> {
    let old = object.remove("max_tokens");
    let new = object.remove("max_completion_tokens");
    let selected = new.or(old);
    match selected {
        Some(value) => value
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| "Chat Completions max token count must be a positive integer".to_owned()),
        None => Ok(4096),
    }
}

/// Validate the small set of legacy-accepted fields whose semantics are not
/// represented by Anthropic Messages, then drop them from the upstream body.
fn validate_ignored_legacy_fields(object: &mut Map<String, Value>) -> Result<(), String> {
    if let Some(user) = object.remove("user")
        && !user.is_string()
    {
        return Err("Chat Completions user must be a string".to_owned());
    }
    if let Some(n) = object.remove("n")
        && n.as_u64().is_none_or(|value| value == 0)
    {
        return Err("Chat Completions n must be a positive integer".to_owned());
    }
    if let Some(seed) = object.remove("seed")
        && seed.as_i64().is_none()
    {
        return Err("Chat Completions seed must be an integer".to_owned());
    }
    if let Some(parallel) = object.remove("parallel_tool_calls")
        && !parallel.is_boolean()
    {
        return Err("Chat Completions parallel_tool_calls must be a boolean".to_owned());
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Role-specific mapping is kept in one exhaustive match"
)]
/// Translate Chat Completions messages into Anthropic role and content blocks.
fn translate_messages(value: &Value) -> Result<(Vec<Value>, Option<String>), String> {
    let messages = value.as_array().ok_or("Chat Completions messages must be an array")?;
    if messages.is_empty() {
        return Err("Chat Completions messages must not be empty".to_owned());
    }
    let mut output = Vec::with_capacity(messages.len());
    let mut system = Vec::new();
    for message in messages {
        let object = message
            .as_object()
            .ok_or("each Chat Completions message must be an object")?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .ok_or("each Chat Completions message requires a string role")?;
        let allowed: &[&str] = match role {
            "system" | "developer" | "user" => &["role", "content"],
            "assistant" => &["role", "content", "tool_calls"],
            "tool" => &["role", "content", "tool_call_id"],
            _ => &[],
        };
        if let Some(field) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(format!(
                "unsupported Chat Completions message field {field:?} cannot be preserved by Anthropic Messages"
            ));
        }
        match role {
            "system" | "developer" => system.push(text_content(object.get("content"))?),
            "user" => output.push(json!({"role":"user","content": translate_content(object.get("content"))?})),
            "assistant" => {
                let mut content = Vec::new();
                let has_tool_calls = object.contains_key("tool_calls");
                if let Some(value) = object
                    .get("content")
                    .filter(|value| !(value.is_null() || has_tool_calls && value.as_str().is_some_and(str::is_empty)))
                {
                    let blocks = translate_content(Some(value))?;
                    if blocks
                        .iter()
                        .any(|block| block.get("type").and_then(Value::as_str) == Some("image"))
                    {
                        return Err("image_url content is supported only in user messages".to_owned());
                    }
                    content.extend(blocks);
                }
                if let Some(calls) = object.get("tool_calls") {
                    content.extend(translate_tool_calls(calls)?);
                }
                if content.is_empty() {
                    return Err("assistant message must contain content or tool_calls".to_owned());
                }
                output.push(json!({"role":"assistant","content":content}));
            },
            "tool" => {
                let call_id = object
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or("tool message requires tool_call_id")?;
                let result = text_content(object.get("content"))?;
                output.push(
                    json!({"role":"user","content":[{"type":"tool_result","tool_use_id":call_id,"content":result}]}),
                );
            },
            other => return Err(format!("unsupported Chat Completions message role {other:?}")),
        }
    }
    if output.is_empty() {
        return Err("request must contain at least one user or assistant message".to_owned());
    }
    Ok((output, (!system.is_empty()).then(|| system.join("\n"))))
}

/// Translate a string or supported text/image content blocks.
fn translate_content(value: Option<&Value>) -> Result<Vec<Value>, String> {
    let value = value.ok_or("Chat Completions message requires content")?;
    if let Some(text) = value.as_str() {
        return Ok(vec![json!({"type":"text","text":text})]);
    }
    let blocks = value
        .as_array()
        .ok_or("Chat Completions content must be a string or an array of text blocks")?;
    blocks
        .iter()
        .map(|block| {
            let object = block
                .as_object()
                .ok_or("Chat Completions content block must be an object")?;
            match object.get("type").and_then(Value::as_str) {
                Some("text") => {
                    reject_nested_fields(object, &["type", "text"], "text content block")?;
                    let text = object
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or("text content block requires a string text")?;
                    Ok(json!({"type":"text","text":text}))
                },
                Some("image_url") => translate_image_url(object),
                Some(kind) => Err(format!("unsupported Chat Completions content block {kind:?}")),
                None => Err("Chat Completions content block requires type".to_owned()),
            }
        })
        .collect()
}

/// Convert an OpenAI `image_url` content block to an Anthropic image source.
/// Remote URLs are passed through as URL references; Praxis never fetches them.
fn translate_image_url(object: &Map<String, Value>) -> Result<Value, String> {
    reject_nested_fields(object, &["type", "image_url"], "image_url content block")?;
    let image = object
        .get("image_url")
        .and_then(Value::as_object)
        .ok_or("image_url content block requires an image_url object")?;
    reject_nested_fields(image, &["url", "detail"], "image_url")?;
    if let Some(detail) = image.get("detail")
        && !matches!(detail.as_str(), Some("auto" | "low" | "high"))
    {
        return Err("image_url detail must be auto, low, or high".to_owned());
    }
    let source = image
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .ok_or("image_url requires a non-empty string url")?;
    if let Some(data_url) = source.strip_prefix("data:") {
        return translate_base64_data_url(data_url);
    }
    let url = Url::parse(source).map_err(|_error| "image_url must be a valid HTTPS or base64 data URL")?;
    if url.scheme() != "https" || url.host_str().is_none() || !url.username().is_empty() || url.password().is_some() {
        return Err("remote image_url must be an HTTPS URL without embedded credentials".to_owned());
    }
    Ok(json!({"type":"image","source":{"type":"url","url":source}}))
}

/// Convert a validated base64 data URL to an Anthropic image block.
fn translate_base64_data_url(value: &str) -> Result<Value, String> {
    let (metadata, encoded) = value
        .split_once(',')
        .ok_or("image data URL must contain a base64 payload")?;
    let mut metadata_parts = metadata.split(';');
    let media_type = metadata_parts.next().unwrap_or_default();
    if !matches!(media_type, "image/jpeg" | "image/png" | "image/gif" | "image/webp") {
        return Err("image data URL media type must be JPEG, PNG, GIF, or WebP".to_owned());
    }
    if metadata_parts.next() != Some("base64") || metadata_parts.next().is_some() {
        return Err("image data URL must use the base64 encoding marker".to_owned());
    }
    let mut decoder = DecoderReader::new(encoded.as_bytes(), &STANDARD);
    // The longest signature is 12 bytes; validate the rest without retaining the image.
    let mut signature = Vec::with_capacity(12);
    decoder
        .by_ref()
        .take(12)
        .read_to_end(&mut signature)
        .map_err(|_error| "image data URL contains invalid base64")?;
    std::io::copy(&mut decoder, &mut std::io::sink()).map_err(|_error| "image data URL contains invalid base64")?;
    if signature.is_empty() {
        return Err("image data URL payload must not be empty".to_owned());
    }
    if !matches_image_signature(media_type, &signature) {
        return Err("image data URL payload does not match its declared image media type".to_owned());
    }
    Ok(json!({"type":"image","source":{"type":"base64","media_type":media_type,"data":encoded}}))
}

/// Check that the decoded image begins with the signature for its declared type.
fn matches_image_signature(media_type: &str, bytes: &[u8]) -> bool {
    match media_type {
        "image/jpeg" => bytes.starts_with(&[0xFF, 0xD8, 0xFF]),
        "image/png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "image/gif" => bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a"),
        "image/webp" => bytes.len() >= 12 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(&b"WEBP"[..]),
        _ => false,
    }
}

/// Flatten validated text blocks for Anthropic system and tool-result content.
fn text_content(value: Option<&Value>) -> Result<String, String> {
    let blocks = translate_content(value)?;
    let parts = blocks
        .iter()
        .map(|block| {
            block
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| "invalid text block".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(parts.concat())
}

/// Convert assistant function calls into Anthropic `tool_use` blocks.
fn translate_tool_calls(value: &Value) -> Result<Vec<Value>, String> {
    value
        .as_array()
        .ok_or("assistant tool_calls must be an array")?
        .iter()
        .map(translate_tool_call)
        .collect()
}

/// Convert one OpenAI function call into an Anthropic `tool_use` block.
fn translate_tool_call(value: &Value) -> Result<Value, String> {
    let call = value.as_object().ok_or("tool call must be an object")?;
    reject_nested_fields(call, &["id", "type", "function"], "tool call")?;
    if call.get("type").and_then(Value::as_str) != Some("function") {
        return Err("only function tool calls can be translated to Anthropic Messages".to_owned());
    }
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("tool call requires id")?;
    let function = call
        .get("function")
        .and_then(Value::as_object)
        .ok_or("tool call requires function")?;
    reject_nested_fields(function, &["name", "arguments"], "tool call function")?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("tool call requires function.name")?;
    let raw = function
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or("tool call arguments must be a JSON string")?;
    let input: Value =
        serde_json::from_str(raw).map_err(|error| format!("tool call arguments are invalid JSON: {error}"))?;
    if !input.is_object() {
        return Err("tool call arguments must decode to a JSON object".to_owned());
    }
    Ok(json!({"type":"tool_use","id":id,"name":name,"input":input}))
}

/// Convert function tool schemas into Anthropic `tools` entries.
fn translate_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools.iter().map(translate_tool).collect()
}

/// Convert one OpenAI function tool definition into an Anthropic tool.
fn translate_tool(value: &Value) -> Result<Value, String> {
    let tool = value.as_object().ok_or("tool must be an object")?;
    reject_nested_fields(tool, &["type", "function"], "tool")?;
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return Err("only function tools can be translated to Anthropic Messages".to_owned());
    }
    let function = tool
        .get("function")
        .and_then(Value::as_object)
        .ok_or("function tool requires function object")?;
    reject_nested_fields(function, &["name", "description", "parameters"], "function tool")?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or("function tool requires name")?;
    let schema = match function.get("parameters") {
        Some(schema) if schema.is_object() => schema.clone(),
        Some(_) => return Err("function tool parameters must be a JSON object".to_owned()),
        None => json!({"type":"object","properties":{}}),
    };
    let mut translated = Map::new();
    translated.insert("name".to_owned(), Value::String(name.to_owned()));
    translated.insert("input_schema".to_owned(), schema);
    if let Some(description) = function.get("description").and_then(Value::as_str) {
        translated.insert("description".to_owned(), Value::String(description.to_owned()));
    }
    Ok(Value::Object(translated))
}

/// Convert OpenAI tool selection into the corresponding Anthropic choice.
fn translate_tool_choice(value: &Value) -> Result<Option<Value>, String> {
    if let Some(choice) = value.as_str() {
        return match choice {
            "auto" => Ok(Some(json!({"type":"auto"}))),
            "required" => Ok(Some(json!({"type":"any"}))),
            "none" => Ok(Some(json!({"type":"none"}))),
            other => Err(format!("unsupported tool_choice value {other:?}")),
        };
    }
    let object = value
        .as_object()
        .ok_or("tool_choice must be a string or function object")?;
    reject_nested_fields(object, &["type", "function"], "tool_choice")?;
    match object.get("type").and_then(Value::as_str) {
        Some("function") => {
            let name = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or("named tool_choice requires function.name")?;
            Ok(Some(json!({"type":"tool","name":name})))
        },
        Some(kind) => Err(format!("unsupported tool_choice type {kind:?}")),
        None => Err("tool_choice function object requires type".to_owned()),
    }
}

/// Identify OpenAI's no-tools choice, which does not require translated tools.
fn is_tool_choice_none(value: &Value) -> bool {
    value.as_str() == Some("none")
}

/// Normalize one or more OpenAI stop strings to Anthropic's sequence array.
fn normalize_stop(value: Value) -> Result<Value, String> {
    match value {
        Value::String(value) => Ok(json!([value])),
        Value::Array(values) if values.iter().all(Value::is_string) => Ok(Value::Array(values)),
        _ => Err("Chat Completions stop must be a string or an array of strings".to_owned()),
    }
}

/// Extract the OpenAI streaming usage preference from `stream_options`.
fn include_usage_from(object: &mut Map<String, Value>) -> Result<bool, String> {
    let Some(value) = object.remove("stream_options") else {
        return Ok(false);
    };
    let options = value.as_object().ok_or("stream_options must be an object")?;
    if options.keys().any(|key| key != "include_usage") {
        return Err("unsupported stream_options field".to_owned());
    }
    options.get("include_usage").map_or(Ok(false), |value| {
        value
            .as_bool()
            .ok_or_else(|| "stream_options.include_usage must be a boolean".to_owned())
    })
}

/// Remove a required string member from a JSON object.
fn take_string(object: &mut Map<String, Value>, key: &str) -> Result<String, String> {
    object
        .remove(key)
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| format!("Chat Completions request requires string {key}"))
}

/// Reject unrecognized members of nested request objects.
fn reject_nested_fields(object: &Map<String, Value>, allowed: &[&str], context: &str) -> Result<(), String> {
    if let Some(field) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!(
            "unsupported {context} field {field:?} cannot be preserved by Anthropic Messages"
        ));
    }
    Ok(())
}
