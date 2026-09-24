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
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

/// Translate an OpenAI Chat Completions request body into Bedrock Converse
/// JSON, and return the translated bytes alongside the model name and stream
/// flag.
///
/// Returns `Err(message)` when the body is not valid JSON, is not a JSON
/// object, or is missing the required `model` field.
pub(crate) fn transform_request(body: &[u8]) -> Result<TransformResult, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;

    let obj = value.as_object().ok_or("request body is not a JSON object")?;

    // Extract model — required; becomes the URL path segment.
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or("missing required field `model`")?
        .to_owned();
    validate_model_id_for_path(&model)?;

    // Extract stream flag before translation.
    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let mut converse: Map<String, Value> = Map::new();

    // Collect system/developer prompts into the top-level `system` array.
    let system_parts = collect_system_parts(obj)?;
    if !system_parts.is_empty() {
        converse.insert("system".to_owned(), Value::Array(system_parts));
    }

    // Translate the message array.
    let messages = translate_messages(obj)?;
    converse.insert("messages".to_owned(), Value::Array(messages));

    // Inference parameters.
    let inference_config = build_inference_config(obj);
    if !inference_config.is_empty() {
        converse.insert("inferenceConfig".to_owned(), Value::Object(inference_config));
    }

    // Tool configuration.
    if let Some(tool_config) = build_tool_config(obj)? {
        converse.insert("toolConfig".to_owned(), tool_config);
    }

    let body_bytes = serde_json::to_vec(&Value::Object(converse)).map_err(|e| format!("serialization failed: {e}"))?;

    Ok(TransformResult {
        body: body_bytes,
        model,
        stream,
    })
}

/// Validate a Bedrock model ID for safe upstream path construction.
fn validate_model_id_for_path(model: &str) -> Result<(), String> {
    if model.is_empty() || model.len() > MAX_MODEL_ID_LEN {
        return Err(format!(
            "field `model` must be between 1 and {MAX_MODEL_ID_LEN} characters"
        ));
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
// System message extraction
// -----------------------------------------------------------------------------

/// Collect all `system` and `developer` role messages from the Chat
/// Completions message array into Bedrock `system` content blocks.
///
/// Bedrock places system context in a top-level `system` array rather
/// than inline in the `messages` array.
fn collect_system_parts(obj: &Map<String, Value>) -> Result<Vec<Value>, String> {
    let Some(messages) = obj.get("messages").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut parts = Vec::new();
    for msg in messages {
        let Some(role) = msg.get("role").and_then(Value::as_str) else {
            continue;
        };
        if role == "system" || role == "developer" {
            let content = msg
                .get("content")
                .ok_or_else(|| format!("{role} message is missing `content`"))?;
            append_system_content(&mut parts, content, role)?;
        }
    }
    Ok(parts)
}

/// Append one system/developer message without dropping multipart text.
fn append_system_content(parts: &mut Vec<Value>, content: &Value, role: &str) -> Result<(), String> {
    match content {
        Value::String(text) => parts.push(serde_json::json!({"text": text})),
        Value::Array(content_parts) => {
            for part in content_parts {
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
                if part_type != "text" {
                    return Err(format!("unsupported {role} content part type `{part_type}`"));
                }
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{role} text content part is missing `text`"))?;
                parts.push(serde_json::json!({"text": text}));
            }
        },
        _ => return Err(format!("{role} message `content` must be a string or array")),
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Message translation
// -----------------------------------------------------------------------------

/// Translate the Chat Completions `messages` array into Bedrock Converse
/// `messages`, skipping `system` and `developer` roles (handled separately).
#[expect(
    clippy::too_many_lines,
    reason = "dispatches all user content-part types; splitting would spread related logic across functions"
)]
fn translate_messages(obj: &Map<String, Value>) -> Result<Vec<Value>, String> {
    let Some(messages) = obj.get("messages").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or_default();

        match role {
            "system" | "developer" => {
                // Already extracted into the top-level `system` field.
            },
            "user" => {
                let content = translate_user_content(msg)?;
                out.push(serde_json::json!({
                    "role": "user",
                    "content": content
                }));
            },
            "assistant" => {
                let content = translate_assistant_content(msg)?;
                out.push(serde_json::json!({
                    "role": "assistant",
                    "content": content
                }));
            },
            "tool" => {
                // Bedrock requires tool results inside a user-role message.
                // Merge consecutive tool messages into one to preserve the
                // required user/assistant role alternation.
                let block = translate_tool_result(msg)?;
                // Immutable check first so the borrow ends before the
                // mutable access or push below.
                let merge = out.last().is_some_and(|last| {
                    last.get("role").and_then(Value::as_str) == Some("user")
                        && last
                            .get("content")
                            .and_then(Value::as_array)
                            .is_some_and(|c| c.iter().all(|b| b.get("toolResult").is_some()))
                });
                if merge {
                    if let Some(arr) = out
                        .last_mut()
                        .and_then(|m| m.get_mut("content"))
                        .and_then(Value::as_array_mut)
                    {
                        arr.push(block);
                    }
                } else {
                    out.push(serde_json::json!({
                        "role": "user",
                        "content": [block]
                    }));
                }
            },
            other => return Err(format!("unsupported message role `{other}`")),
        }
    }

    Ok(out)
}

// -----------------------------------------------------------------------------
// Per-role content translators
// -----------------------------------------------------------------------------

/// Translate a `user` message's content into Bedrock content blocks.
///
/// Handles both string content and the multipart content-part array.
fn translate_user_content(msg: &Value) -> Result<Vec<Value>, String> {
    let content = msg.get("content");
    match content {
        Some(Value::String(s)) => Ok(vec![serde_json::json!({"text": s})]),
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::new();
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
                match part_type {
                    "text" => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            blocks.push(serde_json::json!({"text": text}));
                        }
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

/// Translate an `assistant` message's content into Bedrock content blocks.
///
/// An assistant message may contain:
/// - Plain text in `content`
/// - `tool_calls` requesting function invocations
/// - Both simultaneously (text + tool calls)
#[expect(
    clippy::too_many_lines,
    reason = "maps both text and tool-call content blocks; splitting would obscure the logic"
)]
fn translate_assistant_content(msg: &Value) -> Result<Vec<Value>, String> {
    let mut blocks: Vec<Value> = Vec::new();

    // Text content (may be null when the message is tool-calls only).
    match msg.get("content") {
        Some(Value::String(s)) if !s.is_empty() => {
            blocks.push(serde_json::json!({"text": s}));
        },
        Some(Value::Array(parts)) => {
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
                if part_type != "text" {
                    return Err(format!("unsupported assistant content part type `{part_type}`"));
                }
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or("assistant text content part is missing `text`")?;
                if !text.is_empty() {
                    blocks.push(serde_json::json!({"text": text}));
                }
            }
        },
        _ => {},
    }

    // Tool calls → `toolUse` blocks.
    if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            let id = tc
                .get("id")
                .and_then(Value::as_str)
                .ok_or("tool call is missing `id`")?;
            let function = tc.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .ok_or("tool call is missing `function.name`")?;
            // Parse the JSON-encoded arguments string into a Value so Bedrock
            // receives a nested object, not a JSON string.
            let arguments = function
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .ok_or("tool call is missing `function.arguments`")?;
            let input: Value = serde_json::from_str(arguments)
                .map_err(|e| format!("tool call `function.arguments` is not valid JSON: {e}"))?;

            blocks.push(serde_json::json!({
                "toolUse": {
                    "toolUseId": id,
                    "name": name,
                    "input": input
                }
            }));
        }
    }

    // Bedrock requires at least one content block.
    if blocks.is_empty() {
        blocks.push(serde_json::json!({"text": ""}));
    }

    Ok(blocks)
}

/// Translate a `tool` role message into a Bedrock `toolResult` content block.
///
/// Returns a single `{"toolResult": {...}}` block; the caller places it into
/// a `user`-role message, merging with adjacent blocks when appropriate.
fn translate_tool_result(msg: &Value) -> Result<Value, String> {
    let tool_use_id = msg
        .get("tool_call_id")
        .and_then(Value::as_str)
        .ok_or("tool message is missing `tool_call_id`")?;

    // The tool result content can be a string or structured JSON.
    let content_block = match msg.get("content") {
        Some(Value::String(s)) => vec![serde_json::json!({"text": s})],
        Some(v) if !v.is_null() => {
            // Pass structured content through as JSON text.
            vec![serde_json::json!({"json": v})]
        },
        _ => vec![serde_json::json!({"text": ""})],
    };

    Ok(serde_json::json!({
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": content_block
        }
    }))
}

// -----------------------------------------------------------------------------
// Inference configuration
// -----------------------------------------------------------------------------

/// Build Bedrock `inferenceConfig` from Chat Completions generation params.
///
/// Only parameters in the Converse base set are mapped.  Provider-specific
/// parameters (e.g. `top_k`) are silently dropped; operators can add them
/// via `additionalModelRequestFields` at the Bedrock gateway level.
fn build_inference_config(obj: &Map<String, Value>) -> Map<String, Value> {
    let mut cfg = Map::new();

    // max_tokens / max_completion_tokens → maxTokens
    let max_tokens = obj.get("max_tokens").or_else(|| obj.get("max_completion_tokens"));
    if let Some(n) = max_tokens.and_then(Value::as_u64) {
        cfg.insert("maxTokens".to_owned(), Value::Number(serde_json::Number::from(n)));
    }

    // temperature → temperature
    if let Some(t) = obj.get("temperature").and_then(Value::as_f64)
        && let Some(n) = serde_json::Number::from_f64(t)
    {
        cfg.insert("temperature".to_owned(), Value::Number(n));
    }

    // top_p → topP
    if let Some(p) = obj.get("top_p").and_then(Value::as_f64)
        && let Some(n) = serde_json::Number::from_f64(p)
    {
        cfg.insert("topP".to_owned(), Value::Number(n));
    }

    // stop → stopSequences (string or array)
    let stop_seqs = match obj.get("stop") {
        Some(Value::String(s)) => vec![Value::String(s.clone())],
        Some(Value::Array(arr)) => arr.iter().filter(|v| v.is_string()).cloned().collect(),
        _ => Vec::new(),
    };
    if !stop_seqs.is_empty() {
        cfg.insert("stopSequences".to_owned(), Value::Array(stop_seqs));
    }

    cfg
}

// -----------------------------------------------------------------------------
// Tool configuration
// -----------------------------------------------------------------------------

/// Build the Bedrock `toolConfig` from Chat Completions `tools` and
/// `tool_choice` fields.
///
/// Returns `None` if no tools are defined.
fn build_tool_config(obj: &Map<String, Value>) -> Result<Option<Value>, String> {
    if obj.get("tool_choice").and_then(Value::as_str) == Some("none") {
        return Ok(None);
    }

    let Some(tools) = obj.get("tools") else {
        return Ok(None);
    };
    let tools_arr = tools.as_array().ok_or("`tools` must be an array")?;
    if tools_arr.is_empty() {
        return Ok(None);
    }

    let bedrock_tools: Vec<Value> = tools_arr.iter().map(translate_tool_spec).collect::<Result<_, _>>()?;

    let mut tool_config = serde_json::json!({"tools": bedrock_tools});

    if let Some(choice) = translate_tool_choice(obj)
        && let Some(obj) = tool_config.as_object_mut()
    {
        obj.insert("toolChoice".to_owned(), choice);
    }

    Ok(Some(tool_config))
}

/// Convert a single Chat Completions tool definition to Bedrock `toolSpec`.
fn translate_tool_spec(tool: &Value) -> Result<Value, String> {
    // We only handle `function` type tools.
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return Err("only function tools are supported".to_owned());
    }
    let func = tool.get("function").ok_or("function tool is missing `function`")?;
    let name = func
        .get("name")
        .and_then(Value::as_str)
        .ok_or("function tool is missing `function.name`")?;
    let description = func.get("description").and_then(Value::as_str).unwrap_or("");
    let schema = func
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));

    Ok(serde_json::json!({
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {"json": schema}
        }
    }))
}

/// Translate Chat Completions `tool_choice` into a Bedrock `toolChoice` object.
///
/// | Chat Completions          | Bedrock                          |
/// |---------------------------|----------------------------------|
/// | `"none"`                  | entire `toolConfig` omitted      |
/// | `"auto"` / absent         | `{"auto": {}}`                   |
/// | `"required"`              | `{"any": {}}`                    |
/// | `{"type":"function","function":{"name":"f"}}` | `{"tool":{"name":"f"}}` |
fn translate_tool_choice(obj: &Map<String, Value>) -> Option<Value> {
    match obj.get("tool_choice") {
        Some(Value::String(s)) if s == "required" => Some(serde_json::json!({"any": {}})),
        Some(Value::String(s)) if s == "none" => None,
        Some(Value::Object(tc)) => {
            let name = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str)?;
            Some(serde_json::json!({"tool": {"name": name}}))
        },
        _ => Some(serde_json::json!({"auto": {}})),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
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
    fn user_string_content_wrapped_in_text_block() {
        let body = translate(r#"{"model":"m","messages":[{"role":"user","content":"Hello!"}]}"#);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"][0]["text"], "Hello!");
    }

    #[test]
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

    /// A `tool` message following a plain `user` message must produce a new
    /// `user` message rather than appending its `toolResult` block to the
    /// preceding one (which contains only text, not `toolResult` blocks).
    #[test]
    fn tool_message_after_plain_user_is_not_merged() {
        let body = translate(
            r#"{"model":"m","messages":[
                {"role":"user","content":"here is some context"},
                {"role":"tool","tool_call_id":"t1","content":"result"}
            ]}"#,
        );
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(
            msgs.len(),
            2,
            "tool result must not be merged into the preceding plain user message"
        );
        assert_eq!(msgs[0]["content"][0]["text"], "here is some context");
        assert!(msgs[1]["content"][0].get("toolResult").is_some());
    }

    /// Parallel tool calls produce multiple back-to-back `tool` messages.
    /// They must be merged into a single Bedrock `user` message so the output
    /// sequence maintains strict user/assistant alternation.
    #[test]
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
    fn max_tokens_maps_to_max_tokens() {
        let body = translate(r#"{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(body["inferenceConfig"]["maxTokens"], 512);
    }

    #[test]
    fn temperature_and_top_p_mapped() {
        let body =
            translate(r#"{"model":"m","temperature":0.7,"top_p":0.9,"messages":[{"role":"user","content":"hi"}]}"#);
        let cfg = &body["inferenceConfig"];
        assert!((cfg["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-9);
        assert!((cfg["topP"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn stop_string_wrapped_in_array() {
        let body = translate(r#"{"model":"m","stop":"\n\n","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(body["inferenceConfig"]["stopSequences"][0], "\n\n");
    }

    #[test]
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
    fn tool_choice_auto_emits_auto_object() {
        let body = translate(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice":"auto"}"#,
        );
        assert!(body["toolConfig"]["toolChoice"]["auto"].is_object());
    }

    #[test]
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
    fn empty_messages_produces_empty_array() {
        let body = translate(r#"{"model":"m","messages":[]}"#);
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }
}
