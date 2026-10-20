// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request translation: OpenAI Chat Completions → AWS Bedrock Converse.
//!
//! Converts a Chat Completions request body into the shape expected by
//! `POST /model/{modelId}/converse` (or `/converse-stream`).
//!
//! ## Field mapping
//!
//! | Chat Completions field | Bedrock Converse field              |
//! |------------------------|-------------------------------------|
//! | `model`                | removed from body; becomes URL path |
//! | `messages[].role`      | `messages[].role` (user/assistant)  |
//! | `messages[].content`   | `messages[].content[].text`         |
//! | `system` / `developer` | top-level `system[].text`           |
//! | `tool` role messages   | user role + `toolResult` block      |
//! | `tool_calls` in asst.  | `toolUse` content blocks            |
//! | `tools`                | `toolConfig.tools[].toolSpec`       |
//! | `tool_choice`          | `toolConfig.toolChoice`             |
//! | `max_tokens`           | `inferenceConfig.maxTokens`         |
//! | `temperature`          | `inferenceConfig.temperature`       |
//! | `top_p`                | `inferenceConfig.topP`              |
//! | `stop`                 | `inferenceConfig.stopSequences`     |
//! | `stream`               | removed; determines endpoint        |
//! | `stream_options`       | controls streaming usage output     |

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

/// Maximum Bedrock model ID length.
const MAX_MODEL_ID_LEN: usize = 2048;

/// Bedrock Converse `modelId` pattern.
const MODEL_ID_PATTERN: &str = concat!(
    r"\A(?:",
    r"arn:aws(?:-[a-z0-9-]+)?:bedrock:[a-z0-9-]{1,20}:(?:",
    r"[0-9]{12}:custom-model/[a-z0-9-]{1,63}\.[a-z0-9-]{1,63}/[a-z0-9]{12}|",
    r":foundation-model/[a-z0-9-]{1,63}\.[a-z0-9-]{1,63}(?:[.:]?[a-z0-9-]{1,63})|",
    r"[0-9]{12}:imported-model/[a-z0-9]{12}|",
    r"[0-9]{12}:provisioned-model/[a-z0-9]{12}|",
    r"[0-9]{12}:custom-model-deployment/[a-z0-9]{12}|",
    r"[0-9]{12}:(?:inference-profile|application-inference-profile)/[a-zA-Z0-9-:.]+",
    r")|",
    r"[a-z0-9-]{1,63}\.[a-z0-9-]{1,63}(?:[.:]?[a-z0-9-]{1,63})|",
    r"(?:[0-9a-zA-Z][_-]?)+|",
    r"[a-zA-Z0-9-:.]+|",
    r"arn:aws(?:-[a-z0-9-]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:prompt/[0-9a-zA-Z]{10}(?::[0-9]{1,5})?|",
    r"arn:aws:sagemaker:[a-z0-9-]+:[0-9]{12}:endpoint/[a-zA-Z0-9-]+|",
    r"arn:aws(?:-[a-z0-9-]+)?:bedrock:[0-9a-z-]{1,20}:[0-9]{12}:(?:default-)?prompt-router/[a-zA-Z0-9-:.]+",
    r")\z",
);

/// Compiled Bedrock model ID pattern.
static MODEL_ID_REGEX: LazyLock<Result<Regex, regex::Error>> = LazyLock::new(|| Regex::new(MODEL_ID_PATTERN));

// -----------------------------------------------------------------------------
// Public result type
// -----------------------------------------------------------------------------

/// Result of a successful request translation.
#[derive(Debug)]
pub(crate) struct TransformResult {
    /// The translated Bedrock Converse request body (JSON bytes).
    pub body: Vec<u8>,
    /// Model identifier extracted from the original request body,
    /// used to build the upstream URL path.
    pub model: String,
    /// Whether the original request asked for streaming output.
    pub stream: bool,
    /// Whether the caller requested the final streaming usage chunk.
    pub include_usage: bool,
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

/// Translate an OpenAI Chat Completions request body into Bedrock Converse
/// JSON, and return the translated bytes alongside the model and streaming
/// options.
///
/// Returns `Err(message)` when the request is malformed or cannot be
/// represented by Bedrock Converse.
pub(crate) fn transform_request(body: &[u8]) -> Result<TransformResult, String> {
    let mut obj: Map<String, Value> = match serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))? {
        Value::Object(map) => map,
        _ => return Err("request body is not a JSON object".to_owned()),
    };

    obj.get("messages")
        .and_then(Value::as_array)
        .ok_or("missing required array field `messages`")?;
    let stream = optional_bool(&obj, "stream")?.unwrap_or(false);
    let include_usage = validate_request_semantics(&obj, stream)?;

    let model = extract_model_id(&obj)?;

    let mut converse: Map<String, Value> = Map::new();

    // Split the message array in one consuming pass: system/developer
    // content is hoisted to the top-level `system` array and the remaining
    // roles become Converse `messages`, moving values instead of copying.
    let (system_parts, translated) = translate_messages(&mut obj)?;
    if !system_parts.is_empty() {
        converse.insert("system".to_owned(), Value::Array(system_parts));
    }
    converse.insert("messages".to_owned(), Value::Array(translated));

    // Inference parameters.
    let inference_config = build_inference_config(&mut obj)?;
    if !inference_config.is_empty() {
        converse.insert("inferenceConfig".to_owned(), Value::Object(inference_config));
    }

    // Tool configuration.
    if let Some(tool_config) = build_tool_config(&mut obj)? {
        converse.insert("toolConfig".to_owned(), tool_config);
    }

    let body_bytes = serde_json::to_vec(&Value::Object(converse)).map_err(|e| format!("serialization failed: {e}"))?;

    Ok(TransformResult {
        body: body_bytes,
        model,
        stream,
        include_usage,
    })
}

// -----------------------------------------------------------------------------
// Request validation
// -----------------------------------------------------------------------------

/// Validate request semantics and return whether streaming usage was requested.
fn validate_request_semantics(obj: &Map<String, Value>, stream: bool) -> Result<bool, String> {
    let include_usage = validate_stream_options(obj, stream)?;

    for (field, value) in obj {
        if is_supported_request_field(field) || value.is_null() || is_noop_request_field(field, value) {
            continue;
        }
        return Err(format!("`{field}` is not supported by the Bedrock Converse translator"));
    }
    drop(translate_tool_choice(obj)?);
    Ok(include_usage)
}

/// Return whether the translator consumes a top-level request field.
fn is_supported_request_field(field: &str) -> bool {
    matches!(
        field,
        "model"
            | "messages"
            | "stream"
            | "stream_options"
            | "max_tokens"
            | "max_completion_tokens"
            | "temperature"
            | "top_p"
            | "stop"
            | "tools"
            | "tool_choice"
    )
}

/// Validate streaming options and return whether usage was requested.
fn validate_stream_options(obj: &Map<String, Value>, stream: bool) -> Result<bool, String> {
    let Some(value) = obj.get("stream_options") else {
        return Ok(false);
    };
    if value.is_null() {
        return Ok(false);
    }
    if !stream {
        return Err("`stream_options` requires `stream: true`".to_owned());
    }

    let options = value.as_object().ok_or("`stream_options` must be an object")?;
    let mut include_usage = false;
    for (field, value) in options {
        match field.as_str() {
            "include_usage" => {
                include_usage = value
                    .as_bool()
                    .ok_or("`stream_options.include_usage` must be a boolean")?;
            },
            other => {
                return Err(format!(
                    "`stream_options.{other}` is not supported by the Bedrock Converse translator"
                ));
            },
        }
    }

    Ok(include_usage)
}

/// Read a nullable optional boolean without silently coercing other types.
fn optional_bool(obj: &Map<String, Value>, field: &str) -> Result<Option<bool>, String> {
    match obj.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("`{field}` must be a boolean")),
    }
}

/// Return whether an unsupported field carries its documented no-op default.
fn is_noop_request_field(field: &str, value: &Value) -> bool {
    match field {
        "n" => *value == 1,
        "frequency_penalty" | "presence_penalty" => *value == 0.0,
        "logprobs" | "store" => *value == false,
        "modalities" => *value == serde_json::json!(["text"]),
        "response_format" => *value == serde_json::json!({"type": "text"}),
        "parallel_tool_calls" => *value == true,
        "logit_bias" | "metadata" => value.as_object().is_some_and(Map::is_empty),
        "service_tier" => *value == "auto",
        "verbosity" => *value == "medium",
        _ => false,
    }
}

/// Extract and validate the model ID used in the upstream path.
fn extract_model_id(obj: &Map<String, Value>) -> Result<String, String> {
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or("missing required field `model`")?
        .to_owned();
    validate_model_id_for_path(&model)?;
    Ok(model)
}

/// Validate a Bedrock model ID for safe upstream path construction.
fn validate_model_id_for_path(model: &str) -> Result<(), String> {
    if model.is_empty() || model.len() > MAX_MODEL_ID_LEN {
        return Err(format!(
            "field `model` must be between 1 and {MAX_MODEL_ID_LEN} characters"
        ));
    }
    if model == "." || model == ".." {
        return Err("field `model` is not a valid Bedrock model ID".to_owned());
    }

    let pattern = MODEL_ID_REGEX
        .as_ref()
        .map_err(|_error| "Bedrock model ID validation is unavailable".to_owned())?;
    if !pattern.is_match(model) {
        return Err("field `model` is not a valid Bedrock model ID".to_owned());
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Message translation
// -----------------------------------------------------------------------------

/// Translate the Chat Completions `messages` array into Bedrock Converse
/// `messages`, hoisting `system` and `developer` roles into a top-level
/// `system` parts array.
///
/// The messages are consumed in place: text payloads are moved into the
/// translated blocks rather than copied. System/developer content is
/// extracted first so malformed system content is reported before any
/// other message translation error, matching the original two-pass order.
#[expect(
    clippy::too_many_lines,
    reason = "two passes over one array (system hoisting, then role translation); splitting would duplicate the walking logic"
)]
fn translate_messages(obj: &mut Map<String, Value>) -> Result<(Vec<Value>, Vec<Value>), String> {
    let Value::Array(mut messages) = obj
        .remove("messages")
        .ok_or("missing required array field `messages`")?
    else {
        return Err("missing required array field `messages`".to_owned());
    };
    let mut system_parts = Vec::new();

    // Pass 1: hoist system/developer content, taking each payload by value.
    for msg in &mut messages {
        let is_system = msg.get("role").and_then(Value::as_str) == Some("system");
        let is_developer = msg.get("role").and_then(Value::as_str) == Some("developer");
        if !is_system && !is_developer {
            continue;
        }
        let role = if is_developer { "developer" } else { "system" };
        let Some(content) = msg.get_mut("content") else {
            return Err(format!("{role} message is missing `content`"));
        };
        append_system_content(&mut system_parts, std::mem::take(content), role)?;
    }

    // Pass 2: translate the remaining roles.
    let mut out = Vec::new();
    for msg in &mut messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or_default();

        match role {
            "system" | "developer" => {
                // Already extracted into the top-level `system` field.
            },
            "user" => {
                let content = msg.get_mut("content").map(std::mem::take);
                let blocks = translate_user_content(content)?;
                push_or_merge_message(&mut out, "user", blocks);
            },
            "assistant" => {
                let blocks = translate_assistant_content(msg)?;
                push_or_merge_message(&mut out, "assistant", blocks);
            },
            "tool" => {
                // Bedrock requires tool results inside a user-role message.
                let block = translate_tool_result(msg)?;
                push_or_merge_message(&mut out, "user", vec![block]);
            },
            other => return Err(format!("unsupported message role `{other}`")),
        }
    }

    Ok((system_parts, out))
}

/// Preserve Bedrock's strict role alternation by combining adjacent messages
/// with the same translated role while retaining content-block order.
fn push_or_merge_message(out: &mut Vec<Value>, role: &str, mut blocks: Vec<Value>) {
    if let Some(content) = out
        .last_mut()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some(role))
        .and_then(|message| message.get_mut("content"))
        .and_then(Value::as_array_mut)
    {
        content.append(&mut blocks);
        return;
    }

    out.push(serde_json::json!({"role": role, "content": blocks}));
}

// -----------------------------------------------------------------------------
// Per-role content translators
// -----------------------------------------------------------------------------

/// Build a `{"text": …}` content block, moving the payload string into the
/// block instead of copying it.
fn text_block(text: String) -> Value {
    let mut block = Map::with_capacity(1);
    block.insert("text".to_owned(), Value::String(text));
    Value::Object(block)
}

/// Append one system/developer message without dropping multipart text,
/// taking the content by value so text payloads move into `parts`.
fn append_system_content(parts: &mut Vec<Value>, content: Value, role: &str) -> Result<(), String> {
    match content {
        Value::String(text) => parts.push(text_block(text)),
        Value::Array(content_parts) => {
            for part in content_parts {
                let Value::Object(mut part) = part else {
                    return Err(format!("unsupported {role} content part type ``"));
                };
                let part_type = match part.remove("type") {
                    Some(Value::String(part_type)) => part_type,
                    _ => String::new(),
                };
                if part_type != "text" {
                    return Err(format!("unsupported {role} content part type `{part_type}`"));
                }
                let Some(Value::String(text)) = part.remove("text") else {
                    return Err(format!("{role} text content part is missing `text`"));
                };
                parts.push(text_block(text));
            }
        },
        _ => return Err(format!("{role} message `content` must be a string or array")),
    }
    Ok(())
}

/// Translate a `user` message's content into Bedrock content blocks,
/// moving text payloads instead of copying them.
///
/// Handles both string content and the multipart content-part array.
#[expect(
    clippy::too_many_lines,
    reason = "validates both supported user content shapes while building the output blocks"
)]
fn translate_user_content(content: Option<Value>) -> Result<Vec<Value>, String> {
    match content {
        Some(Value::String(s)) => Ok(vec![text_block(s)]),
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::new();
            for part in parts {
                let Value::Object(mut part) = part else {
                    return Err("unsupported user content part type ``".to_owned());
                };
                let part_type = match part.remove("type") {
                    Some(Value::String(part_type)) => part_type,
                    _ => String::new(),
                };
                match part_type.as_str() {
                    "text" => {
                        let Some(Value::String(text)) = part.remove("text") else {
                            return Err("user text content part is missing `text`".to_owned());
                        };
                        blocks.push(text_block(text));
                    },
                    "image_url" => {
                        return Err("`image_url` content parts are not supported by this translator".to_owned());
                    },
                    other => {
                        return Err(format!("unsupported user content part type `{other}`"));
                    },
                }
            }
            if blocks.is_empty() {
                // Bedrock requires at least one content block per message.
                return Err("user message produced no translatable content blocks".to_owned());
            }
            Ok(blocks)
        },
        _ => Err("user message `content` must be a string or array".to_owned()),
    }
}

/// Translate an `assistant` message's content into Bedrock content blocks,
/// moving text and tool-call payloads instead of copying them.
///
/// An assistant message may contain:
/// - Plain text in `content`
/// - `tool_calls` requesting function invocations
/// - Both simultaneously (text + tool calls)
#[expect(
    clippy::too_many_lines,
    reason = "maps both text and tool-call content blocks; splitting would obscure the logic"
)]
fn translate_assistant_content(msg: &mut Value) -> Result<Vec<Value>, String> {
    let mut blocks: Vec<Value> = Vec::new();

    // Text content (may be null when the message is tool-calls only).
    match msg.get_mut("content").map(std::mem::take) {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                blocks.push(text_block(s));
            }
        },
        Some(Value::Array(parts)) => {
            for part in parts {
                let Value::Object(mut part) = part else {
                    return Err("unsupported assistant content part type ``".to_owned());
                };
                let part_type = match part.remove("type") {
                    Some(Value::String(part_type)) => part_type,
                    _ => String::new(),
                };
                if part_type != "text" {
                    return Err(format!("unsupported assistant content part type `{part_type}`"));
                }
                let Some(Value::String(text)) = part.remove("text") else {
                    return Err("assistant text content part is missing `text`".to_owned());
                };
                if !text.is_empty() {
                    blocks.push(text_block(text));
                }
            }
        },
        None | Some(Value::Null) => {},
        _ => return Err("assistant message `content` must be a string, array, or null".to_owned()),
    }

    // Tool calls → `toolUse` blocks.
    if let Some(tool_calls) = msg.get_mut("tool_calls").map(std::mem::take)
        && !tool_calls.is_null()
    {
        let Value::Array(tool_calls) = tool_calls else {
            return Err("assistant message `tool_calls` must be an array".to_owned());
        };
        for tc in tool_calls {
            let Value::Object(mut tc) = tc else {
                return Err("tool call is missing `id`".to_owned());
            };
            let Some(Value::String(id)) = tc.remove("id") else {
                return Err("tool call is missing `id`".to_owned());
            };
            let Some(Value::Object(mut function)) = tc.remove("function") else {
                return Err("tool call is missing `function.name`".to_owned());
            };
            let Some(Value::String(name)) = function.remove("name") else {
                return Err("tool call is missing `function.name`".to_owned());
            };
            // Parse the JSON-encoded arguments string into a Value so Bedrock
            // receives a nested object, not a JSON string.
            let Some(Value::String(arguments)) = function.remove("arguments") else {
                return Err("tool call is missing `function.arguments`".to_owned());
            };
            let input: Value = serde_json::from_str(&arguments)
                .map_err(|e| format!("tool call `function.arguments` is not valid JSON: {e}"))?;

            let mut tool_use = Map::with_capacity(3);
            tool_use.insert("toolUseId".to_owned(), Value::String(id));
            tool_use.insert("name".to_owned(), Value::String(name));
            tool_use.insert("input".to_owned(), input);
            let mut block = Map::with_capacity(1);
            block.insert("toolUse".to_owned(), Value::Object(tool_use));
            blocks.push(Value::Object(block));
        }
    }

    // Bedrock requires at least one content block.
    if blocks.is_empty() {
        blocks.push(text_block(String::new()));
    }

    Ok(blocks)
}

/// Translate a `tool` role message into a Bedrock `toolResult` content block,
/// moving the result text instead of copying it.
///
/// Returns a single `{"toolResult": {…}}` block; the caller places it into
/// a `user`-role message, merging with adjacent blocks when appropriate.
#[expect(
    clippy::too_many_lines,
    reason = "validates both supported OpenAI tool-result content shapes before building one Bedrock block"
)]
fn translate_tool_result(msg: &mut Value) -> Result<Value, String> {
    let Some(Value::String(tool_use_id)) = msg.get_mut("tool_call_id").map(std::mem::take) else {
        return Err("tool message is missing `tool_call_id`".to_owned());
    };

    // OpenAI tool results support a string or multipart text content.
    let content_block = match msg.get_mut("content").map(std::mem::take) {
        Some(Value::String(s)) => vec![text_block(s)],
        Some(Value::Array(parts)) => parts
            .into_iter()
            .map(|part| {
                let Value::Object(mut part) = part else {
                    return Err("unsupported tool content part type ``".to_owned());
                };
                let part_type = match part.remove("type") {
                    Some(Value::String(part_type)) => part_type,
                    _ => String::new(),
                };
                if part_type != "text" {
                    return Err(format!("unsupported tool content part type `{part_type}`"));
                }
                let Some(Value::String(text)) = part.remove("text") else {
                    return Err("tool text content part is missing `text`".to_owned());
                };
                Ok(text_block(text))
            })
            .collect::<Result<Vec<_>, String>>()?,
        Some(value) if !value.is_null() => {
            return Err("tool message `content` must be a string or array".to_owned());
        },
        _ => vec![text_block(String::new())],
    };

    let mut result = Map::with_capacity(2);
    result.insert("toolUseId".to_owned(), Value::String(tool_use_id));
    result.insert("content".to_owned(), Value::Array(content_block));
    let mut block = Map::with_capacity(1);
    block.insert("toolResult".to_owned(), Value::Object(result));
    Ok(Value::Object(block))
}

// -----------------------------------------------------------------------------
// Inference configuration
// -----------------------------------------------------------------------------

/// Build Bedrock `inferenceConfig` from Chat Completions generation params.
///
/// Only parameters in the Converse base set are mapped. Unsupported fields
/// are rejected before this function is called.
#[expect(
    clippy::too_many_lines,
    reason = "maps the complete Bedrock inferenceConfig field set in one linear validation pass"
)]
fn build_inference_config(obj: &mut Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut cfg = Map::new();

    // max_tokens / max_completion_tokens → maxTokens
    let max_tokens = obj
        .get("max_tokens")
        .filter(|value| !value.is_null())
        .or_else(|| obj.get("max_completion_tokens").filter(|value| !value.is_null()));
    if let Some(value) = max_tokens {
        let n = value
            .as_u64()
            .ok_or("`max_tokens` and `max_completion_tokens` must be non-negative integers")?;
        cfg.insert("maxTokens".to_owned(), Value::Number(serde_json::Number::from(n)));
    }

    // temperature → temperature
    if let Some(value) = obj.get("temperature").filter(|value| !value.is_null()) {
        let t = value.as_f64().ok_or("`temperature` must be a number")?;
        let n = serde_json::Number::from_f64(t).ok_or("`temperature` must be a finite number")?;
        cfg.insert("temperature".to_owned(), Value::Number(n));
    }

    // top_p → topP
    if let Some(value) = obj.get("top_p").filter(|value| !value.is_null()) {
        let p = value.as_f64().ok_or("`top_p` must be a number")?;
        let n = serde_json::Number::from_f64(p).ok_or("`top_p` must be a finite number")?;
        cfg.insert("topP".to_owned(), Value::Number(n));
    }

    // stop → stopSequences (string or array); the array moves element-wise.
    let stop_seqs = match obj.remove("stop") {
        Some(Value::String(s)) => vec![Value::String(s)],
        Some(Value::Array(values)) => {
            if values.iter().any(|value| !value.is_string()) {
                return Err("`stop` array values must be strings".to_owned());
            }
            values
        },
        None | Some(Value::Null) => Vec::new(),
        Some(_) => return Err("`stop` must be a string or array of strings".to_owned()),
    };
    if !stop_seqs.is_empty() {
        cfg.insert("stopSequences".to_owned(), Value::Array(stop_seqs));
    }

    Ok(cfg)
}

// -----------------------------------------------------------------------------
// Tool configuration
// -----------------------------------------------------------------------------

/// Build the Bedrock `toolConfig` from Chat Completions `tools` and
/// `tool_choice` fields.
///
/// Returns `None` if no tools are defined.
fn build_tool_config(obj: &mut Map<String, Value>) -> Result<Option<Value>, String> {
    if obj.get("tool_choice").and_then(Value::as_str) == Some("none") {
        return Ok(None);
    }

    let tools_arr = match obj.remove("tools") {
        Some(Value::Array(tools)) => tools,
        Some(_) => return Err("`tools` must be an array".to_owned()),
        None => return Ok(None),
    };
    if tools_arr.is_empty() {
        return Ok(None);
    }

    let bedrock_tools: Vec<Value> = tools_arr
        .into_iter()
        .map(translate_tool_spec)
        .collect::<Result<_, _>>()?;

    let mut tool_config = serde_json::json!({"tools": bedrock_tools});

    if let Some(choice) = translate_tool_choice(obj)?
        && let Some(obj) = tool_config.as_object_mut()
    {
        obj.insert("toolChoice".to_owned(), choice);
    }

    Ok(Some(tool_config))
}

/// Convert a single Chat Completions tool definition to Bedrock `toolSpec`,
/// moving the input schema and function metadata instead of copying them.
#[expect(
    clippy::too_many_lines,
    reason = "validates the function tool shape field by field while moving each field into the toolSpec"
)]
fn translate_tool_spec(tool: Value) -> Result<Value, String> {
    // We only handle `function` type tools.
    let Value::Object(mut tool) = tool else {
        return Err("only function tools are supported".to_owned());
    };
    let tool_type = match tool.remove("type") {
        Some(Value::String(tool_type)) => tool_type,
        _ => String::new(),
    };
    if tool_type != "function" {
        return Err("only function tools are supported".to_owned());
    }
    let Some(Value::Object(mut func)) = tool.remove("function") else {
        return Err("function tool is missing object `function`".to_owned());
    };
    let Some(Value::String(name)) = func.remove("name") else {
        return Err("function tool is missing `function.name`".to_owned());
    };
    let description =
        take_optional_string_field(&mut func, "description")?.filter(|description| !description.is_empty());
    let strict = take_optional_bool_field(&mut func, "strict")?;
    let schema = func
        .remove("parameters")
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));

    let mut tool_spec = Map::with_capacity(4);
    tool_spec.insert("name".to_owned(), Value::String(name));
    let mut input_schema = Map::with_capacity(1);
    input_schema.insert("json".to_owned(), schema);
    tool_spec.insert("inputSchema".to_owned(), Value::Object(input_schema));
    if let Some(description) = description {
        tool_spec.insert("description".to_owned(), Value::String(description));
    }
    if let Some(strict) = strict {
        tool_spec.insert("strict".to_owned(), Value::Bool(strict));
    }

    let mut outer = Map::with_capacity(1);
    outer.insert("toolSpec".to_owned(), Value::Object(tool_spec));
    Ok(Value::Object(outer))
}

/// Remove an optional string function field while rejecting a wrong JSON type.
fn take_optional_string_field(obj: &mut Map<String, Value>, field: &str) -> Result<Option<String>, String> {
    match obj.remove(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(format!("`function.{field}` must be a string or null")),
    }
}

/// Remove an optional boolean function field while rejecting a wrong JSON type.
fn take_optional_bool_field(obj: &mut Map<String, Value>, field: &str) -> Result<Option<bool>, String> {
    match obj.remove(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(value)),
        Some(_) => Err(format!("`function.{field}` must be a boolean or null")),
    }
}

/// Translate Chat Completions `tool_choice` into a Bedrock `toolChoice` object.
///
/// | Chat Completions          | Bedrock                          |
/// |---------------------------|----------------------------------|
/// | `"none"`                  | entire `toolConfig` omitted      |
/// | `"auto"` / absent         | `{"auto": {}}`                   |
/// | `"required"`              | `{"any": {}}`                    |
/// | `{"type":"function","function":{"name":"f"}}` | `{"tool":{"name":"f"}}` |
fn translate_tool_choice(obj: &Map<String, Value>) -> Result<Option<Value>, String> {
    match obj.get("tool_choice") {
        None | Some(Value::Null) => Ok(Some(serde_json::json!({"auto": {}}))),
        Some(Value::String(s)) if s == "auto" => Ok(Some(serde_json::json!({"auto": {}}))),
        Some(Value::String(s)) if s == "required" => Ok(Some(serde_json::json!({"any": {}}))),
        Some(Value::String(s)) if s == "none" => Ok(None),
        Some(Value::String(s)) => Err(format!("unsupported `tool_choice` value `{s}`")),
        Some(Value::Object(tc)) => {
            if tc.get("type").and_then(Value::as_str) != Some("function") {
                return Err("object `tool_choice` must have type `function`".to_owned());
            }
            let name = tc
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or("object `tool_choice` is missing `function.name`")?;
            Ok(Some(serde_json::json!({"tool": {"name": name}})))
        },
        Some(_) => Err("`tool_choice` must be a string or object".to_owned()),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::Value;

    use super::*;

    fn translate(json: &str) -> Value {
        let result = transform_request(json.as_bytes()).unwrap();
        serde_json::from_slice(&result.body).unwrap()
    }

    fn translate_result(json: &str) -> TransformResult {
        transform_request(json.as_bytes()).unwrap()
    }

    // ── Model extraction ──────────────────────────────────────────────────

    #[test]
    fn model_extracted_not_in_body() {
        let req = r#"{"model":"anthropic.claude-3-sonnet-20240229-v1:0","messages":[{"role":"user","content":"hi"}]}"#;
        let result = translate_result(req);
        assert_eq!(result.model, "anthropic.claude-3-sonnet-20240229-v1:0");
        let body: Value = serde_json::from_slice(&result.body).unwrap();
        assert!(body.get("model").is_none(), "model must not appear in body");
    }

    #[test]
    fn missing_model_returns_error() {
        let err = transform_request(br#"{"messages":[]}"#).unwrap_err();
        assert!(err.contains("model"), "error must mention `model`");
    }

    #[test]
    fn valid_model_id_forms_are_accepted() {
        let models = [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/profile.name:1",
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile.name:1",
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/name.name/abcdefghijkl",
            "arn:aws:bedrock:us-east-1:123456789012:imported-model/abcdefghijkl",
            "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abcdefghijkl",
            "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/abcdefghijkl",
            "arn:aws:bedrock:us-east-1:123456789012:prompt/abcdefghij:123",
            "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/router.name:1",
            "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:prompt/abcdefghij",
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint",
        ];

        for model in models {
            let request = format!(r#"{{"model":"{model}","messages":[]}}"#);
            assert_eq!(transform_request(request.as_bytes()).unwrap().model, model);
        }
    }

    #[test]
    fn unsafe_model_ids_are_rejected() {
        let models = [
            "",
            ".",
            "..",
            "../../../other-endpoint",
            "model?target=other",
            "model#fragment",
            "model%2Fother",
            "model/other",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/../other",
            "arn:aws:bedrock:us-east-1:123456789012:unknown-resource/abcdefghijkl",
            "arn:aws:bedrock:us-east-1:123:provisioned-model/abcdefghijkl",
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my_endpoint",
            "arn:aws-?:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
            "arn:aws-#:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
            "arn:aws-%2f:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
            "arn:aws-../..:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
        ];

        for model in models {
            let request = format!(r#"{{"model":"{model}","messages":[]}}"#);
            let error = transform_request(request.as_bytes()).unwrap_err();
            assert!(error.contains("model"), "unexpected error for {model}: {error}");
        }
    }

    #[test]
    fn model_id_length_is_validated() {
        let maximum = "a".repeat(MAX_MODEL_ID_LEN);
        let request = format!(r#"{{"model":"{maximum}","messages":[]}}"#);
        assert_eq!(transform_request(request.as_bytes()).unwrap().model, maximum);

        let over_limit = "a".repeat(MAX_MODEL_ID_LEN + 1);
        let request = format!(r#"{{"model":"{over_limit}","messages":[]}}"#);
        let error = transform_request(request.as_bytes()).unwrap_err();
        assert!(error.contains("between 1 and 2048"));
    }

    // ── Request semantics ────────────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn supported_request_fields_are_mapped() {
        let body = translate(
            r#"{"model":"m","messages":[],"max_completion_tokens":128,
                "temperature":0.4,"top_p":0.8,"stop":["done"]}"#,
        );

        assert_eq!(body["inferenceConfig"]["maxTokens"], 128);
        assert_eq!(body["inferenceConfig"]["temperature"], 0.4);
        assert_eq!(body["inferenceConfig"]["topP"], 0.8);
        assert_eq!(body["inferenceConfig"]["stopSequences"], serde_json::json!(["done"]));
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn null_max_tokens_uses_max_completion_tokens_fallback() {
        let body = translate(r#"{"model":"m","messages":[],"max_tokens":null,"max_completion_tokens":128}"#);
        assert_eq!(body["inferenceConfig"]["maxTokens"], 128);
    }

    #[test]
    fn supported_request_fields_with_invalid_types_are_rejected() {
        let fields = [
            ("stream", serde_json::json!("true")),
            ("max_tokens", serde_json::json!("128")),
            ("max_completion_tokens", serde_json::json!(-1)),
            ("temperature", serde_json::json!("0.4")),
            ("top_p", serde_json::json!(true)),
            ("stop", serde_json::json!(["done", 1])),
            ("tool_choice", serde_json::json!(42)),
        ];

        for (field, value) in fields {
            let request = serde_json::json!({
                "model": "m",
                "messages": [],
                field: value,
            });
            let error = transform_request(request.to_string().as_bytes()).unwrap_err();
            assert!(error.contains(field), "rejection must name `{field}`: {error}");
        }
    }

    #[test]
    fn messages_must_be_an_array() {
        for request in [
            serde_json::json!({"model": "m"}),
            serde_json::json!({"model": "m", "messages": {}}),
        ] {
            let error = transform_request(request.to_string().as_bytes()).unwrap_err();
            assert!(error.contains("messages"));
        }
    }

    #[test]
    fn unsupported_request_semantics_are_rejected() {
        let fields = [
            ("n", serde_json::json!(2)),
            ("response_format", serde_json::json!({"type": "json_object"})),
            ("frequency_penalty", serde_json::json!(0.5)),
            ("presence_penalty", serde_json::json!(-0.5)),
            ("logprobs", serde_json::json!(true)),
            ("top_logprobs", serde_json::json!(2)),
            ("seed", serde_json::json!(42)),
            ("parallel_tool_calls", serde_json::json!(false)),
            ("store", serde_json::json!(true)),
            ("modalities", serde_json::json!(["text", "audio"])),
            ("logit_bias", serde_json::json!({"42": 1})),
            ("reasoning_effort", serde_json::json!("high")),
            ("verbosity", serde_json::json!("low")),
            ("prediction", serde_json::json!({"type": "content", "content": "x"})),
            ("top_k", serde_json::json!(40)),
        ];

        for (field, value) in fields {
            let request = serde_json::json!({
                "model": "m",
                "messages": [],
                field: value,
            });
            let error = transform_request(request.to_string().as_bytes()).unwrap_err();
            assert!(error.contains(field), "rejection must name `{field}`: {error}");
        }
    }

    #[test]
    fn unsupported_noop_defaults_are_accepted() {
        let request = serde_json::json!({
            "model": "m",
            "messages": [],
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "logprobs": false,
            "store": false,
            "modalities": ["text"],
            "response_format": {"type": "text"},
            "parallel_tool_calls": true,
            "logit_bias": {},
            "metadata": {},
            "service_tier": "auto",
            "verbosity": "medium",
            "seed": null,
        });

        assert!(transform_request(request.to_string().as_bytes()).is_ok());
    }

    #[test]
    fn stream_usage_option_is_supported_for_streaming_requests() {
        let request = serde_json::json!({
            "model": "m",
            "messages": [],
            "stream": true,
            "stream_options": {"include_usage": true},
        });

        let result = transform_request(request.to_string().as_bytes()).unwrap();
        assert!(result.stream);
        assert!(result.include_usage);
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn stream_usage_false_and_omitted_are_disabled() {
        for stream_options in [None, Some(serde_json::json!({"include_usage": false}))] {
            let mut request = serde_json::json!({
                "model": "m",
                "messages": [],
                "stream": true,
            });
            if let Some(options) = stream_options {
                request["stream_options"] = options;
            }

            let result = transform_request(request.to_string().as_bytes()).unwrap();
            assert!(!result.include_usage);
        }
    }

    #[test]
    fn stream_usage_option_requires_streaming() {
        let request = serde_json::json!({
            "model": "m",
            "messages": [],
            "stream_options": {"include_usage": true},
        });

        let error = transform_request(request.to_string().as_bytes()).unwrap_err();
        assert!(error.contains("stream_options"));
    }

    #[test]
    fn unsupported_stream_option_is_rejected_explicitly() {
        let request = serde_json::json!({
            "model": "m",
            "messages": [],
            "stream": true,
            "stream_options": {"include_obfuscation": true},
        });

        let error = transform_request(request.to_string().as_bytes()).unwrap_err();
        assert!(error.contains("stream_options.include_obfuscation"));
    }

    #[test]
    fn stream_usage_option_requires_a_boolean() {
        let request = serde_json::json!({
            "model": "m",
            "messages": [],
            "stream": true,
            "stream_options": {"include_usage": "true"},
        });

        let error = transform_request(request.to_string().as_bytes()).unwrap_err();
        assert!(error.contains("stream_options.include_usage"));
    }

    #[test]
    fn reasoning_effort_medium_is_not_a_noop() {
        let request = serde_json::json!({
            "model": "m",
            "messages": [],
            "reasoning_effort": "medium",
        });

        let error = transform_request(request.to_string().as_bytes()).unwrap_err();
        assert!(error.contains("reasoning_effort"));
    }

    // ── Stream flag ───────────────────────────────────────────────────────

    #[test]
    fn stream_flag_extracted_true() {
        let result = translate_result(r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(result.stream);
        let body: Value = serde_json::from_slice(&result.body).unwrap();
        assert!(body.get("stream").is_none(), "stream must not be in body");
    }

    #[test]
    fn stream_flag_defaults_to_false() {
        let result = translate_result(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert!(!result.stream);
    }

    // ── System message extraction ─────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn system_role_hoisted_to_top_level() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"system","content":"You are helpful."},
                {"role":"user","content":"Hello"}
            ]}"#,
        );
        let system = &body["system"];
        assert_eq!(system[0]["text"], "You are helpful.");
        // system role must NOT appear in messages
        let msgs = body["messages"].as_array().unwrap();
        assert!(!msgs.iter().any(|m| m["role"] == "system"));
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn developer_role_hoisted_same_as_system() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"developer","content":"Be concise."},
                {"role":"user","content":"Hi"}
            ]}"#,
        );
        assert_eq!(body["system"][0]["text"], "Be concise.");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn all_multipart_system_text_is_preserved() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"system","content":[
                    {"type":"text","text":"First."},
                    {"type":"text","text":"Second."}
                ]},
                {"role":"user","content":"Hi"}
            ]}"#,
        );
        assert_eq!(body["system"][0]["text"], "First.");
        assert_eq!(body["system"][1]["text"], "Second.");
    }

    #[test]
    fn no_system_message_omits_system_field() {
        let body = translate(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert!(body.get("system").is_none());
    }

    // ── User message ──────────────────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn user_string_content_wrapped_in_text_block() {
        let body = translate(r#"{"model":"m","messages":[{"role":"user","content":"Hello!"}]}"#);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"][0]["text"], "Hello!");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn user_multipart_text_parts_translated() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"part one"},
                {"type":"text","text":"part two"}
            ]}]}"#,
        );
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "part one");
        assert_eq!(content[1]["text"], "part two");
    }

    #[test]
    fn unsupported_image_is_rejected_instead_of_silently_dropped() {
        let err = transform_request(
            br#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"describe this"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}
            ]}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("image_url"));
    }

    // ── Assistant message ─────────────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn assistant_text_content_wrapped() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"Hello there!"}
            ]}"#,
        );
        let msg = &body["messages"][1];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"][0]["text"], "Hello there!");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn assistant_tool_calls_produce_tool_use_blocks() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"What is the weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function","function":
                        {"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}
                ]}
            ]}"#,
        );
        let msg = &body["messages"][1];
        assert_eq!(msg["role"], "assistant");
        let block = &msg["content"][0]["toolUse"];
        assert_eq!(block["toolUseId"], "call_1");
        assert_eq!(block["name"], "get_weather");
        assert_eq!(block["input"]["city"], "Paris");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn assistant_text_and_tool_calls_both_present() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"Let me check.","tool_calls":[
                    {"id":"tc1","type":"function","function":
                        {"name":"f","arguments":"{}"}}
                ]}
            ]}"#,
        );
        let content = body["messages"][1]["content"].as_array().unwrap();
        // First block is text, second is toolUse.
        assert_eq!(content[0]["text"], "Let me check.");
        assert!(content[1].get("toolUse").is_some());
    }

    #[test]
    fn malformed_tool_arguments_are_rejected_instead_of_replaced() {
        let err = transform_request(
            br#"{"model":"m","messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"tc1","type":"function","function":{"name":"f","arguments":"{"}}
                ]}
            ]}"#,
        )
        .unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    // ── Tool result message ───────────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_role_becomes_user_with_tool_result_block() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_abc","type":"function","function":
                        {"name":"get_weather","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"call_abc","content":"{\"temp\":18}"}
            ]}"#,
        );
        let msg = &body["messages"][2];
        assert_eq!(msg["role"], "user");
        let tr = &msg["content"][0]["toolResult"];
        assert_eq!(tr["toolUseId"], "call_abc");
        assert_eq!(tr["content"][0]["text"], "{\"temp\":18}");
    }

    /// A translated tool result has the Bedrock `user` role, so it must merge
    /// with adjacent user content to preserve strict role alternation.
    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_message_after_plain_user_is_merged() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"here is some context"},
                {"role":"tool","tool_call_id":"t1","content":"result"}
            ]}"#,
        );
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"][0]["text"], "here is some context");
        assert!(msgs[0]["content"][1].get("toolResult").is_some());
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn adjacent_user_and_assistant_messages_are_merged_by_role() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"first"},
                {"role":"user","content":"second"},
                {"role":"assistant","content":"third"},
                {"role":"assistant","content":"fourth"}
            ]}"#,
        );
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[1]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn multipart_tool_result_text_is_preserved() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"t1","type":"function","function":{"name":"f","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"t1","content":[
                    {"type":"text","text":"first"},
                    {"type":"text","text":"second"}
                ]}
            ]}"#,
        );
        let content = body["messages"][1]["content"][0]["toolResult"]["content"]
            .as_array()
            .unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "first");
        assert_eq!(content[1]["text"], "second");
    }

    #[test]
    fn unsupported_multipart_tool_result_part_is_rejected() {
        let error = transform_request(
            br#"{"model":"m","messages":[
                {"role":"tool","tool_call_id":"t1","content":[{"type":"image_url","image_url":{}}]}
            ]}"#,
        )
        .unwrap_err();
        assert!(error.contains("tool content part type"));
    }

    #[test]
    fn malformed_content_and_tool_calls_are_rejected_before_translation() {
        for (request, expected) in [
            (
                serde_json::json!({"role": "user", "content": [{"type": "text", "text": 42}]}),
                "user text",
            ),
            (
                serde_json::json!({"role": "assistant", "content": 42}),
                "assistant message `content`",
            ),
            (
                serde_json::json!({"role": "assistant", "content": null, "tool_calls": "lookup"}),
                "tool_calls",
            ),
        ] {
            let body = serde_json::json!({"model": "m", "messages": [request]});
            let error = transform_request(body.to_string().as_bytes()).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn empty_assistant_text_preserves_tool_calls() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"assistant","content":"","tool_calls":[
                    {"id":"t1","type":"function","function":{"name":"lookup","arguments":"{}"}}
                ]}
            ]}"#,
        );
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["toolUse"]["toolUseId"], "t1");
        assert_eq!(blocks[0]["toolUse"]["name"], "lookup");
    }

    /// Parallel tool calls produce multiple back-to-back `tool` messages.
    /// They must be merged into a single Bedrock `user` message so the output
    /// sequence maintains strict user/assistant alternation.
    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn consecutive_tool_messages_merged_into_single_user_message() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"What's the weather in Paris and London?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},
                    {"id":"c2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\"}"}}
                ]},
                {"role":"tool","tool_call_id":"c1","content":"sunny"},
                {"role":"tool","tool_call_id":"c2","content":"rainy"}
            ]}"#,
        );
        let msgs = body["messages"].as_array().unwrap();
        // Expected output: user / assistant / user — three messages, strict alternation.
        assert_eq!(
            msgs.len(),
            3,
            "consecutive tool messages must collapse into one user message"
        );
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "user");

        let content = msgs[2]["content"].as_array().unwrap();
        assert_eq!(
            content.len(),
            2,
            "both toolResult blocks must appear in one content array"
        );
        assert_eq!(content[0]["toolResult"]["toolUseId"], "c1");
        assert_eq!(content[1]["toolResult"]["toolUseId"], "c2");
    }

    /// `toolResult` blocks must appear in the same order as their source `tool`
    /// messages; content values must be preserved.
    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn consecutive_tool_messages_preserve_order() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"t1","type":"function","function":{"name":"a","arguments":"{}"}},
                    {"id":"t2","type":"function","function":{"name":"b","arguments":"{}"}},
                    {"id":"t3","type":"function","function":{"name":"c","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"t1","content":"first"},
                {"role":"tool","tool_call_id":"t2","content":"second"},
                {"role":"tool","tool_call_id":"t3","content":"third"}
            ]}"#,
        );
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2]["role"], "user");

        let content = msgs[2]["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["toolResult"]["toolUseId"], "t1");
        assert_eq!(content[1]["toolResult"]["toolUseId"], "t2");
        assert_eq!(content[2]["toolResult"]["toolUseId"], "t3");
        assert_eq!(content[0]["toolResult"]["content"][0]["text"], "first");
        assert_eq!(content[1]["toolResult"]["content"][0]["text"], "second");
        assert_eq!(content[2]["toolResult"]["content"][0]["text"], "third");
    }

    /// Tool results from separate agentic rounds (each preceded by its own
    /// assistant message) must not be merged.  The input uses a realistic
    /// two-round sequence: the assistant returns one result, replies, then
    /// issues a second tool call.
    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn non_consecutive_tool_messages_not_merged() {
        // Input:  user → assistant(x1) → tool(x1) → assistant(reply + x2) → tool(x2)
        // Output: user / assistant / user(x1) / assistant / user(x2) — five messages.
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"start"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"x1","type":"function","function":{"name":"f","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"x1","content":"result-one"},
                {"role":"assistant","content":"Got it, now let me also check…","tool_calls":[
                    {"id":"x2","type":"function","function":{"name":"g","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"x2","content":"result-two"}
            ]}"#,
        );
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 5);

        // Verify strict alternation.
        for (i, expected) in ["user", "assistant", "user", "assistant", "user"].iter().enumerate() {
            assert_eq!(msgs[i]["role"], *expected, "role mismatch at index {i}");
        }

        // Each tool round produces its own user message with exactly one block.
        let first_result = &msgs[2];
        assert_eq!(first_result["content"].as_array().unwrap().len(), 1);
        assert_eq!(first_result["content"][0]["toolResult"]["toolUseId"], "x1");

        let second_result = &msgs[4];
        assert_eq!(second_result["content"].as_array().unwrap().len(), 1);
        assert_eq!(second_result["content"][0]["toolResult"]["toolUseId"], "x2");
    }

    // ── Inference config ──────────────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn max_tokens_maps_to_max_tokens() {
        let body = translate(r#"{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(body["inferenceConfig"]["maxTokens"], 512);
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn temperature_and_top_p_mapped() {
        let body =
            translate(r#"{"model":"m","temperature":0.7,"top_p":0.9,"messages":[{"role":"user","content":"hi"}]}"#);
        let cfg = &body["inferenceConfig"];
        assert!((cfg["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-9);
        assert!((cfg["topP"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn stop_string_wrapped_in_array() {
        let body = translate(r#"{"model":"m","stop":"\n\n","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(body["inferenceConfig"]["stopSequences"][0], "\n\n");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn stop_array_preserved() {
        let body = translate(r#"{"model":"m","stop":["END","STOP"],"messages":[{"role":"user","content":"hi"}]}"#);
        let seqs = body["inferenceConfig"]["stopSequences"].as_array().unwrap();
        assert_eq!(seqs.len(), 2);
    }

    #[test]
    fn no_generation_params_omits_inference_config() {
        let body = translate(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        // inferenceConfig may be absent or empty when no params given.
        if let Some(cfg) = body.get("inferenceConfig") {
            assert!(cfg.as_object().is_none_or(Map::is_empty), "should be empty");
        }
    }

    // ── Tool configuration ────────────────────────────────────────────────

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tools_translated_to_tool_spec() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{
                "name":"get_weather",
                "description":"Get weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
            }}]}"#,
        );
        let spec = &body["toolConfig"]["tools"][0]["toolSpec"];
        assert_eq!(spec["name"], "get_weather");
        assert_eq!(spec["description"], "Get weather");
        assert_eq!(spec["inputSchema"]["json"]["type"], "object");
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_without_description_omits_bedrock_description() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
        );
        let spec = &body["toolConfig"]["tools"][0]["toolSpec"];
        assert!(spec.get("description").is_none());
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn strict_tool_schema_is_preserved() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{
                "name":"f","strict":true,"parameters":{"type":"object","additionalProperties":false}
            }}]}"#,
        );
        assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["strict"], true);
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_choice_auto_emits_auto_object() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":"auto"}"#,
        );
        assert!(body["toolConfig"]["toolChoice"]["auto"].is_object());
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_choice_required_emits_any_object() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":"required"}"#,
        );
        assert!(body["toolConfig"]["toolChoice"]["any"].is_object());
    }

    #[test]
    fn tool_choice_none_omits_entire_tool_config() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":"none"}"#,
        );
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_choice_specific_function() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":{"type":"function","function":{"name":"f"}}}"#,
        );
        assert_eq!(body["toolConfig"]["toolChoice"]["tool"]["name"], "f");
    }

    #[test]
    fn no_tools_omits_tool_config() {
        let body = translate(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert!(body.get("toolConfig").is_none());
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn invalid_json_returns_error() {
        let err = transform_request(b"not json").unwrap_err();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn non_object_body_returns_error() {
        let err = transform_request(b"[1,2,3]").unwrap_err();
        assert!(err.contains("not a JSON object"));
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn empty_messages_produces_empty_array() {
        let body = translate(r#"{"model":"m","messages":[]}"#);
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    // ── Allocation evidence (allocation-counter) ───────────────────────────

    /// Pre-ownership `translate_tool_spec`: borrowed the tool and deep-cloned
    /// the input schema plus every metadata string. Kept as the allocation
    /// baseline the consuming translator must beat.
    fn translate_tool_spec_legacy(tool: &Value) -> Result<Value, String> {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err("only function tools are supported".to_owned());
        }
        let func = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or("function tool is missing object `function`")?;
        let name = func
            .get("name")
            .and_then(Value::as_str)
            .ok_or("function tool is missing `function.name`")?;
        let description = func
            .get("description")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty());
        let strict = func.get("strict").and_then(Value::as_bool);
        let schema = func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}));

        let mut tool_spec = Map::new();
        tool_spec.insert("name".to_owned(), Value::String(name.to_owned()));
        tool_spec.insert("inputSchema".to_owned(), serde_json::json!({"json": schema}));
        if let Some(description) = description {
            tool_spec.insert("description".to_owned(), Value::String(description.to_owned()));
        }
        if let Some(strict) = strict {
            tool_spec.insert("strict".to_owned(), Value::Bool(strict));
        }

        Ok(serde_json::json!({"toolSpec": tool_spec}))
    }

    /// Pre-ownership stop extraction: borrowed the request and cloned every
    /// stop sequence into a fresh array. Kept as the allocation baseline.
    fn stop_sequences_legacy(obj: &Map<String, Value>) -> Result<Vec<Value>, String> {
        let stop_seqs = match obj.get("stop") {
            Some(Value::String(s)) => vec![Value::String(s.clone())],
            Some(Value::Array(values)) => {
                if values.iter().any(|value| !value.is_string()) {
                    return Err("`stop` array values must be strings".to_owned());
                }
                values.clone()
            },
            None | Some(Value::Null) => Vec::new(),
            Some(_) => return Err("`stop` must be a string or array of strings".to_owned()),
        };
        Ok(stop_seqs)
    }

    /// Build a function tool whose nested schema carries a `description_bytes`
    /// payload string, the stand-in for a large provider schema.
    fn large_schema_tool(description_bytes: usize) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tool",
                "description": "large schema fixture",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payload": {"type": "string", "description": "s".repeat(description_bytes)}
                    }
                }
            }
        })
    }

    #[test]
    fn tool_spec_schema_moves_without_deep_copy() {
        let tool = large_schema_tool(16 * 1024);
        let owned_input = tool.clone();

        let mut owned_output = None;
        let owned = allocation_counter::measure(|| {
            owned_output = Some(translate_tool_spec(owned_input).unwrap());
        });
        let mut legacy_output = None;
        let legacy = allocation_counter::measure(|| {
            legacy_output = Some(translate_tool_spec_legacy(&tool).unwrap());
        });

        assert_eq!(
            owned_output, legacy_output,
            "moved and cloned tool specs must stay byte-equivalent"
        );
        assert!(
            owned.count_total < legacy.count_total,
            "consuming toolSpec must allocate fewer times than the cloning baseline: moved={} cloned={}",
            owned.count_total,
            legacy.count_total
        );
        assert!(
            owned.bytes_total < legacy.bytes_total,
            "consuming toolSpec must allocate fewer bytes than the cloning baseline: moved={} cloned={}",
            owned.bytes_total,
            legacy.bytes_total
        );
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn tool_spec_allocations_do_not_scale_with_schema_size() {
        let small = large_schema_tool(16 * 1024);
        let large = large_schema_tool(64 * 1024);

        let mut small_output = None;
        let small_allocs = allocation_counter::measure(|| {
            small_output = Some(translate_tool_spec(small).unwrap());
        });
        let mut large_output = None;
        let large_allocs = allocation_counter::measure(|| {
            large_output = Some(translate_tool_spec(large).unwrap());
        });

        let payload_len = |output: Value| {
            output["toolSpec"]["inputSchema"]["json"]["properties"]["payload"]["description"]
                .as_str()
                .unwrap()
                .len()
        };
        assert_eq!(payload_len(small_output.unwrap()), 16 * 1024);
        assert_eq!(payload_len(large_output.unwrap()), 64 * 1024);
        assert_eq!(
            small_allocs.count_total, large_allocs.count_total,
            "toolSpec translation allocations must be independent of schema payload size"
        );
        assert!(
            large_allocs.bytes_total <= small_allocs.bytes_total + 1024,
            "toolSpec translation must not allocate proportionally to schema payload: small={} large={}",
            small_allocs.bytes_total,
            large_allocs.bytes_total
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "builds the fixture, measures both paths, and asserts output equality plus the allocation drop"
    )]
    #[test]
    fn stop_sequences_move_without_element_copy() {
        let stop_request = || {
            serde_json::json!({
                "stop": (0..16)
                    .map(|index| serde_json::json!(format!("stop_{index}_{}", "x".repeat(256))))
                    .collect::<Vec<_>>()
            })
            .as_object()
            .unwrap()
            .clone()
        };
        let mut moved_obj = stop_request();
        let legacy_obj = stop_request();

        let mut moved_cfg = None;
        let moved = allocation_counter::measure(|| {
            moved_cfg = Some(build_inference_config(&mut moved_obj).unwrap());
        });
        let mut legacy_stops = None;
        let legacy = allocation_counter::measure(|| {
            legacy_stops = Some(stop_sequences_legacy(&legacy_obj).unwrap());
        });

        let moved_stops = moved_cfg
            .unwrap()
            .get("stopSequences")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([]));
        assert_eq!(
            moved_stops,
            Value::Array(legacy_stops.unwrap()),
            "moved and cloned stop sequences must stay byte-equivalent"
        );
        assert!(
            moved.bytes_total < legacy.bytes_total,
            "stop array must move element-wise instead of cloning: moved={} cloned={}",
            moved.bytes_total,
            legacy.bytes_total
        );
    }

    #[test]
    fn user_message_text_moves_without_copy() {
        let text = "x".repeat(64 * 1024);
        let moved_input = Value::String(text.clone());
        let legacy_input = Value::String(text);

        let mut moved_blocks = None;
        let moved = allocation_counter::measure(|| {
            moved_blocks = Some(translate_user_content(Some(moved_input)).unwrap());
        });
        let mut legacy_blocks = None;
        let legacy = allocation_counter::measure(|| {
            legacy_blocks = Some(vec![serde_json::json!({
                "text": legacy_input.as_str().unwrap()
            })]);
        });

        assert_eq!(
            moved_blocks, legacy_blocks,
            "moved and cloned text blocks must stay byte-equivalent"
        );
        assert!(
            moved.bytes_total < legacy.bytes_total,
            "user text must move into the block instead of being copied: moved={} cloned={}",
            moved.bytes_total,
            legacy.bytes_total
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "covers the three remaining text-move callers (system, assistant, tool result) with one scale pair each"
    )]
    #[test]
    #[expect(clippy::indexing_slicing, reason = "tests index known fixture shapes")]
    fn system_assistant_and_tool_result_text_allocations_do_not_scale_with_payload() {
        // System/developer text moves into the hoisted `system` parts.
        let measure_system = |text_bytes: usize| {
            let text = "x".repeat(text_bytes);
            let mut parts = Vec::new();
            let mut moved_len = 0;
            let allocs = allocation_counter::measure(|| {
                append_system_content(&mut parts, Value::String(text), "system").unwrap();
                moved_len = parts[0]["text"].as_str().unwrap().len();
            });
            (allocs, moved_len)
        };
        let (small, small_len) = measure_system(16 * 1024);
        let (large, large_len) = measure_system(64 * 1024);
        assert_eq!(small_len, 16 * 1024, "small system payload must land intact");
        assert_eq!(large_len, 64 * 1024, "large system payload must land intact");
        assert_eq!(
            small.count_total, large.count_total,
            "system text allocations must be independent of payload size"
        );
        assert!(
            large.bytes_total <= small.bytes_total + 1024,
            "system text must move into the block instead of being copied: small={} large={}",
            small.bytes_total,
            large.bytes_total
        );

        // Assistant text moves out of the message via `mem::take`.
        let measure_assistant = |text_bytes: usize| {
            let mut msg = serde_json::json!({"role": "assistant", "content": "x".repeat(text_bytes)});
            let mut moved_len = 0;
            let allocs = allocation_counter::measure(|| {
                moved_len = translate_assistant_content(&mut msg).unwrap()[0]["text"]
                    .as_str()
                    .unwrap()
                    .len();
            });
            (allocs, moved_len)
        };
        let (small, small_len) = measure_assistant(16 * 1024);
        let (large, large_len) = measure_assistant(64 * 1024);
        assert_eq!(small_len, 16 * 1024, "small assistant payload must land intact");
        assert_eq!(large_len, 64 * 1024, "large assistant payload must land intact");
        assert_eq!(
            small.count_total, large.count_total,
            "assistant text allocations must be independent of payload size"
        );
        assert!(
            large.bytes_total <= small.bytes_total + 1024,
            "assistant text must move into the block instead of being copied: small={} large={}",
            small.bytes_total,
            large.bytes_total
        );

        // Tool-result text moves into the `toolResult` content block.
        let measure_tool_result = |text_bytes: usize| {
            let mut msg = serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "x".repeat(text_bytes)
            });
            let mut moved_len = 0;
            let allocs = allocation_counter::measure(|| {
                moved_len = translate_tool_result(&mut msg).unwrap()["toolResult"]["content"][0]["text"]
                    .as_str()
                    .unwrap()
                    .len();
            });
            (allocs, moved_len)
        };
        let (small, small_len) = measure_tool_result(16 * 1024);
        let (large, large_len) = measure_tool_result(64 * 1024);
        assert_eq!(small_len, 16 * 1024, "small tool-result payload must land intact");
        assert_eq!(large_len, 64 * 1024, "large tool-result payload must land intact");
        assert_eq!(
            small.count_total, large.count_total,
            "tool-result text allocations must be independent of payload size"
        );
        assert!(
            large.bytes_total <= small.bytes_total + 1024,
            "tool-result text must move into the block instead of being copied: small={} large={}",
            small.bytes_total,
            large.bytes_total
        );
    }
}
