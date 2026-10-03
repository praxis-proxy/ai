// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Chat Completions to Vertex AI Gemini request transformation.
//!
//! Translates the full request body: messages → contents, role mapping,
//! parameter nesting under `generationConfig`, tool definitions, and
//! `stream` flag extraction for URL path selection.

use std::borrow::Cow;

use serde_json::{Map, Value, json};
use tracing::warn;

// -----------------------------------------------------------------------------
// Public Entry Point
// -----------------------------------------------------------------------------

/// Result of a request transformation.
#[derive(Debug)]
pub(crate) struct TransformResult {
    /// Transformed request body bytes in Gemini `generateContent` format.
    pub body: Vec<u8>,
    /// The model name extracted from the original request, used for URL
    /// path construction.
    pub model: String,
    /// Whether the client requested streaming (`stream: true`).
    pub stream: bool,
    /// OpenAI `stream_options.include_usage`. Vertex has no equivalent;
    /// the response path emits a trailing usage chunk when this is set.
    pub include_usage: bool,
    /// Number of candidates expected in a successful response.
    ///
    /// The streaming response path uses this to reject an EOF where only a
    /// subset of the requested candidates reached a terminal finish reason.
    pub candidate_count: u64,
}

/// Transform an OpenAI Chat Completions request body into Vertex AI
/// Gemini `generateContent` format.
///
/// Returns the transformed body, the extracted model name, and the
/// streaming flag. The caller uses `model` and `stream` to construct
/// the Vertex AI endpoint path.
///
/// # Errors
///
/// Returns a human-readable error string when the body is not valid
/// JSON or is missing required fields.
pub(crate) fn transform_request(body: &[u8]) -> Result<TransformResult, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;

    let Some(obj) = value.as_object() else {
        return Err("request body is not a JSON object".to_owned());
    };

    let (model, stream, include_usage, candidate_count) = extract_request_options(obj)?;

    let mut gemini = Map::new();

    // Messages → contents + systemInstruction
    let (contents, system_instruction) = convert_messages(obj)?;
    gemini.insert("contents".to_owned(), Value::Array(contents));
    if let Some(instruction) = system_instruction {
        gemini.insert("systemInstruction".to_owned(), instruction);
    }

    // Parameters → generationConfig
    let gen_config = build_generation_config(obj);
    if !gen_config.is_empty() {
        gemini.insert("generationConfig".to_owned(), Value::Object(gen_config));
    }

    // Tools → tools
    if let Some(tools) = convert_tools(obj) {
        gemini.insert("tools".to_owned(), tools);
    }

    // Tool choice → toolConfig
    if let Some(tool_config) = convert_tool_choice(obj) {
        gemini.insert("toolConfig".to_owned(), tool_config);
    }

    let body = serde_json::to_vec(&Value::Object(gemini)).map_err(|e| format!("serialization failed: {e}"))?;

    Ok(TransformResult {
        body,
        model,
        stream,
        include_usage,
        candidate_count,
    })
}

/// Extract fields that control upstream routing and streaming lifecycle.
fn extract_request_options(obj: &Map<String, Value>) -> Result<(String, bool, bool, u64), String> {
    let model = extract_model(obj)?;
    let stream = extract_stream_flag(obj)?;
    let include_usage = include_usage_requested(obj, stream)?;
    Ok((model, stream, include_usage, expected_candidate_count(obj)))
}

/// Extract and validate the `model` field from the request body.
///
/// The model name is interpolated into the Vertex AI URL path, so it must
/// not contain characters that could produce a malformed or traversable URL.
fn extract_model(obj: &Map<String, Value>) -> Result<String, String> {
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "request body must contain a \"model\" field".to_owned())?;

    let trimmed = model.trim();

    if trimmed.is_empty() {
        return Err("model name must not be empty or whitespace-only".to_owned());
    }

    if trimmed
        .bytes()
        .any(|b| !b.is_ascii_alphanumeric() && b != b'-' && b != b'.' && b != b'_')
    {
        return Err("model name must contain only alphanumeric, '-', '.', or '_' characters".to_owned());
    }

    Ok(trimmed.to_owned())
}

/// Extract and validate the `stream` field from the request body.
///
/// `stream` is proxy-critical: it determines which Vertex endpoint is
/// called and whether the response is SSE. A non-boolean value (e.g.
/// `"stream": "true"`) silently becoming `false` would give the client
/// a full JSON blob instead of the expected SSE stream, which is a hard
/// to debug contract violation. Absent or `null` both mean `false` per
/// the OpenAI specification.
fn extract_stream_flag(obj: &Map<String, Value>) -> Result<bool, String> {
    match obj.get("stream") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err("\"stream\" must be a boolean or null".to_owned()),
    }
}

/// Extract `stream_options.include_usage` from the request.
///
/// Returns `Err` when `stream_options` is used without streaming, is not an
/// object, or contains a non-boolean `include_usage` value.
fn include_usage_requested(obj: &Map<String, Value>, stream: bool) -> Result<bool, String> {
    let stream_options = match obj.get("stream_options") {
        None | Some(Value::Null) => return Ok(false),
        Some(Value::Object(options)) => options,
        Some(_) => return Err("\"stream_options\" must be an object or null".to_owned()),
    };
    if !stream {
        return Err("\"stream_options\" requires \"stream\": true".to_owned());
    }
    let Some(include_usage) = stream_options.get("include_usage") else {
        return Ok(false);
    };
    match include_usage {
        Value::Bool(b) => Ok(*b),
        Value::Null => Ok(false),
        other => {
            let kind = match other {
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
                Value::Bool(_) | Value::Null => unreachable!("handled above"),
            };
            Err(format!(
                "\"stream_options.include_usage\" must be a boolean, got {kind}"
            ))
        },
    }
}

/// Return the number of terminal candidates expected from a successful stream.
fn expected_candidate_count(obj: &Map<String, Value>) -> u64 {
    obj.get("n")
        .and_then(Value::as_u64)
        .filter(|count| *count > 0)
        .unwrap_or(1)
}

// -----------------------------------------------------------------------------
// Message Conversion
// -----------------------------------------------------------------------------

/// Convert OpenAI `messages` array into Gemini `contents` and an
/// optional `systemInstruction`.
///
/// Returns `(contents, system_instruction)`.
///
/// Role mapping:
/// - `system` / `developer` → extracted into `systemInstruction`
/// - `user` → `role: "user"`
/// - `assistant` → `role: "model"`
/// - `tool` / `function` → `role: "user"` with `functionResponse` part
fn convert_messages(obj: &Map<String, Value>) -> Result<(Vec<Value>, Option<Value>), String> {
    let Some(Value::Array(messages)) = obj.get("messages") else {
        return Ok((Vec::new(), None));
    };

    let mut contents = Vec::new();
    let mut system_parts = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        convert_one_message(&mut contents, &mut system_parts, messages, i, msg)?;
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(json!({ "parts": system_parts }))
    };

    Ok((contents, system_instruction))
}

/// Translate a single OpenAI message into `contents` or `system_parts`.
fn convert_one_message(
    contents: &mut Vec<Value>,
    system_parts: &mut Vec<Value>,
    messages: &[Value],
    i: usize,
    msg: &Value,
) -> Result<(), String> {
    let Some(role) = msg.get("role").and_then(Value::as_str) else {
        return Ok(());
    };
    match role {
        "system" | "developer" => collect_system_parts(system_parts, msg),
        "user" => convert_user_message(contents, msg),
        "assistant" => convert_assistant_message(contents, msg)?,
        "tool" | "function" => {
            convert_tool_result(contents, messages, i, msg, is_tool_result_continuation(messages, i));
        },
        _ => warn!(role, "dropping message with unknown role"),
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// System Messages
// -----------------------------------------------------------------------------

/// Collect system/developer message content into `systemInstruction` parts.
fn collect_system_parts(parts: &mut Vec<Value>, msg: &Value) {
    match msg.get("content") {
        Some(Value::String(text)) => {
            if !text.is_empty() {
                parts.push(json!({ "text": text }));
            }
        },
        Some(Value::Array(content_parts)) => {
            for part in content_parts {
                if part.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = part.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    parts.push(json!({ "text": text }));
                }
            }
        },
        _ => {},
    }
}

// -----------------------------------------------------------------------------
// User Messages
// -----------------------------------------------------------------------------

/// Convert a `user` role message to a Gemini content entry.
fn convert_user_message(contents: &mut Vec<Value>, msg: &Value) {
    let parts = match msg.get("content") {
        Some(Value::String(text)) => vec![json!({ "text": text })],
        Some(Value::Array(content_parts)) => convert_multipart_content(content_parts),
        _ => return,
    };

    if !parts.is_empty() {
        contents.push(json!({ "role": "user", "parts": parts }));
    }
}

/// Convert OpenAI multipart content array to Gemini parts.
///
/// Handles `text`, `image_url` (both URLs and base64 data URIs).
fn convert_multipart_content(content_parts: &[Value]) -> Vec<Value> {
    let mut parts = Vec::new();

    for part in content_parts {
        let part_type = part.get("type").and_then(Value::as_str).unwrap_or("<missing>");

        match part_type {
            "text" => {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    parts.push(json!({ "text": text }));
                }
            },
            "image_url" => {
                if let Some(gemini_part) = convert_image_url(part) {
                    parts.push(gemini_part);
                }
            },
            _ => {
                warn!(part_type, "dropping unknown content part type");
            },
        }
    }

    parts
}

/// Convert an OpenAI `image_url` content part to a Gemini `inlineData`
/// or `fileData` part.
///
/// - `data:{mime};base64,{data}` → `inlineData` with `mimeType` + `data`
/// - `https://...` → `fileData` with `mimeType` guessed + `fileUri`
fn convert_image_url(part: &Value) -> Option<Value> {
    let url = part
        .get("image_url")
        .and_then(|v| v.get("url"))
        .and_then(Value::as_str)?;

    if let Some(rest) = url.strip_prefix("data:") {
        let (mime, data) = rest.split_once(";base64,")?;
        Some(json!({
            "inlineData": {
                "mimeType": mime,
                "data": data,
            }
        }))
    } else {
        Some(json!({
            "fileData": {
                "mimeType": guess_mime_type(url),
                "fileUri": url,
            }
        }))
    }
}

/// Best-effort MIME type guess from a URL extension.
fn guess_mime_type(url: &str) -> &'static str {
    let path = url.split('?').next().unwrap_or(url);
    if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".gif") {
        "image/gif"
    } else if path.ends_with(".webp") {
        "image/webp"
    } else {
        "image/jpeg"
    }
}

// -----------------------------------------------------------------------------
// Assistant Messages
// -----------------------------------------------------------------------------

/// Convert an `assistant` role message to a Gemini `model` content entry.
///
/// Handles plain text, tool calls (`functionCall` parts), and any
/// `thoughtSignature` stashed on the tool call as `extra_content`.
fn convert_assistant_message(contents: &mut Vec<Value>, msg: &Value) -> Result<(), String> {
    let mut parts = Vec::new();

    collect_assistant_content(&mut parts, msg.get("content"));

    if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
        for tc in tool_calls {
            if let Some(fc) = convert_tool_call(tc)? {
                parts.push(fc);
            }
        }
    }

    if !parts.is_empty() {
        contents.push(json!({ "role": "model", "parts": parts }));
    }

    Ok(())
}

/// Convert string or array-form assistant content into Gemini text parts.
fn collect_assistant_content(parts: &mut Vec<Value>, content: Option<&Value>) {
    match content {
        Some(Value::String(text)) if !text.is_empty() => parts.push(json!({ "text": text })),
        Some(Value::Array(content_parts)) => {
            for part in content_parts {
                let part_type = part.get("type").and_then(Value::as_str).unwrap_or("<missing>");
                let text = match part_type {
                    "text" => part.get("text").and_then(Value::as_str),
                    "refusal" => part.get("refusal").and_then(Value::as_str),
                    _ => {
                        warn!(part_type, "dropping unknown assistant content part type");
                        None
                    },
                };
                if let Some(text) = text
                    && !text.is_empty()
                {
                    parts.push(json!({ "text": text }));
                }
            }
        },
        _ => {},
    }
}

/// Convert a single OpenAI tool call to a Gemini `functionCall` part.
///
/// Copies `extra_content.google.thought_signature` onto the Part when present
/// (required by Gemini 3 for tool continuations). Returns `Err` when
/// `function.arguments` is absent, not valid JSON, or not a JSON object.
fn convert_tool_call(tc: &Value) -> Result<Option<Value>, String> {
    let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) else {
        return Ok(None);
    };

    let args_str = tc
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .ok_or_else(|| "tool call is missing required field function.arguments".to_owned())?;

    let args: Value =
        serde_json::from_str(args_str).map_err(|e| format!("tool call arguments are not valid JSON: {e}"))?;

    if !args.is_object() {
        return Err(format!(
            "tool call arguments must be a JSON object, got {kind}",
            kind = json_kind(&args),
        ));
    }

    let mut part = json!({
        "functionCall": {
            "name": name,
            "args": args,
        }
    });

    if let Some(sig) = tc.pointer("/extra_content/google/thought_signature")
        && !sig.is_null()
        && let Some(obj) = part.as_object_mut()
    {
        obj.insert("thoughtSignature".to_owned(), sig.clone());
    }

    Ok(Some(part))
}

/// Return the JSON type name of a value for error messages.
fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// -----------------------------------------------------------------------------
// Tool Result Messages
// -----------------------------------------------------------------------------

/// Returns `true` when `messages[index - 1]` is also a `tool`/`function` result.
fn is_tool_result_continuation(messages: &[Value], index: usize) -> bool {
    index > 0
        && messages
            .get(index - 1)
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            .is_some_and(|r| r == "tool" || r == "function")
}

/// Convert a `tool` or `function` role message to a Gemini
/// `functionResponse` part.
///
/// Resolves the function name in order of preference:
/// 1. `msg["name"]` — present on deprecated `function` messages and some clients that add it to `tool` messages for
///    convenience
/// 2. `tool_call_id` → nearest preceding assistant `tool_calls` entry with that id
/// 3. Fallback to `"unknown"` with a warning (Gemini will likely reject this with 400)
///
/// When `append_to_last` is `true`, the part is appended to the last `contents`
/// entry rather than opening a new `user` turn (required for parallel tool calls).
fn convert_tool_result(contents: &mut Vec<Value>, messages: &[Value], index: usize, msg: &Value, append_to_last: bool) {
    let name = resolve_tool_function_name(messages, index, msg);

    let content = tool_result_content(msg.get("content"));

    // Gemini response must be a Struct (JSON object); wrap plain text and
    // valid-but-non-object JSON (arrays, numbers) in {"result": ...}.
    let response: Value = match serde_json::from_str::<Value>(&content) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => json!({ "result": content.as_ref() }),
    };

    let part = json!({
        "functionResponse": {
            "name": name,
            "response": response,
        }
    });

    if append_to_last
        && let Some(last) = contents.last_mut()
        && let Some(Value::Array(parts)) = last.get_mut("parts")
    {
        parts.push(part);
        return;
    }

    contents.push(json!({
        "role": "user",
        "parts": [part]
    }));
}

/// Flatten OpenAI tool-result text parts while borrowing the common string form.
fn tool_result_content(content: Option<&Value>) -> Cow<'_, str> {
    match content {
        Some(Value::String(text)) => Cow::Borrowed(text),
        Some(Value::Array(parts)) => Cow::Owned(
            parts
                .iter()
                .filter_map(|part| {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        part.get("text").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => Cow::Borrowed(""),
    }
}

/// Resolve the function name for a tool/function result message.
///
/// Tries `msg["name"]` first (always present on deprecated `function`
/// messages, sometimes present on `tool` messages). Falls back to the
/// nearest preceding assistant `tool_calls` entry with a matching id,
/// so a reused id from a later round cannot steal an earlier result.
fn resolve_tool_function_name<'a>(messages: &'a [Value], tool_index: usize, msg: &'a Value) -> &'a str {
    if let Some(name) = msg.get("name").and_then(Value::as_str)
        && !name.is_empty()
    {
        return name;
    }

    let Some(tool_call_id) = msg.get("tool_call_id").and_then(Value::as_str) else {
        warn!("tool message has neither name nor tool_call_id");
        return "unknown";
    };

    if let Some(name) = name_from_preceding_tool_call(messages, tool_index, tool_call_id) {
        return name;
    }

    warn!(
        tool_call_id,
        "tool message tool_call_id has no matching assistant tool_call; \
         using \"unknown\" as function name"
    );
    "unknown"
}

/// Scan messages before `tool_index` for the nearest assistant `tool_calls`
/// entry whose `id` matches `tool_call_id`.
fn name_from_preceding_tool_call<'a>(messages: &'a [Value], tool_index: usize, tool_call_id: &str) -> Option<&'a str> {
    let (prior, _) = messages.split_at(tool_index);
    for earlier in prior.iter().rev() {
        if earlier.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(Value::Array(tool_calls)) = earlier.get("tool_calls") else {
            continue;
        };
        for tc in tool_calls.iter().rev() {
            if tc.get("id").and_then(Value::as_str) != Some(tool_call_id) {
                continue;
            }
            if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str)
                && !name.is_empty()
            {
                return Some(name);
            }
        }
    }
    None
}

// -----------------------------------------------------------------------------
// Generation Config
// -----------------------------------------------------------------------------

/// Build the Gemini `generationConfig` object from OpenAI parameters.
///
/// Maps:
/// - `max_tokens` / `max_completion_tokens` → `maxOutputTokens`
/// - `temperature` → `temperature`
/// - `top_p` → `topP`
/// - `stop` → `stopSequences`
/// - `presence_penalty` → `presencePenalty`
/// - `frequency_penalty` → `frequencyPenalty`
/// - `seed` → `seed`
/// - `response_format` → `responseMimeType` + `responseSchema`
fn build_generation_config(obj: &Map<String, Value>) -> Map<String, Value> {
    let mut config = Map::new();

    // max_completion_tokens takes precedence over max_tokens.
    if let Some(v) = obj.get("max_completion_tokens").or_else(|| obj.get("max_tokens")) {
        config.insert("maxOutputTokens".to_owned(), v.clone());
    }

    copy_generation_params(&mut config, obj);
    copy_stop_sequences(&mut config, obj);
    convert_response_format(&mut config, obj);

    config
}

/// Copy simple 1:1 generation parameters from OpenAI to Gemini names.
fn copy_generation_params(config: &mut Map<String, Value>, obj: &Map<String, Value>) {
    const MAPPINGS: &[(&str, &str)] = &[
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("presence_penalty", "presencePenalty"),
        ("frequency_penalty", "frequencyPenalty"),
        ("seed", "seed"),
        ("n", "candidateCount"),
        ("top_logprobs", "logprobs"),
    ];
    for &(openai, gemini) in MAPPINGS {
        if let Some(v) = obj.get(openai) {
            config.insert(gemini.to_owned(), v.clone());
        }
    }
    if obj.get("logprobs") == Some(&Value::Bool(true)) {
        config.insert("responseLogprobs".to_owned(), Value::Bool(true));
    }
}

/// Map `stop` (string or array) to Gemini `stopSequences`.
fn copy_stop_sequences(config: &mut Map<String, Value>, obj: &Map<String, Value>) {
    match obj.get("stop") {
        Some(Value::Array(stops)) => {
            config.insert("stopSequences".to_owned(), Value::Array(stops.clone()));
        },
        Some(Value::String(s)) => {
            config.insert("stopSequences".to_owned(), json!([s]));
        },
        _ => {},
    }
}

/// Convert OpenAI `response_format` to Gemini `responseMimeType` and
/// optionally `responseSchema`.
///
/// - `{"type": "json_object"}` → `responseMimeType: "application/json"`
/// - `{"type": "json_schema", "json_schema": {"schema": ...}}` → `responseMimeType: "application/json"` +
///   `responseSchema`
/// - `{"type": "text"}` → no-op (Gemini default)
fn convert_response_format(config: &mut Map<String, Value>, obj: &Map<String, Value>) {
    let Some(Value::Object(rf)) = obj.get("response_format") else {
        return;
    };

    let format_type = rf.get("type").and_then(Value::as_str).unwrap_or("");

    match format_type {
        "json_object" => {
            config.insert(
                "responseMimeType".to_owned(),
                Value::String("application/json".to_owned()),
            );
        },
        "json_schema" => {
            config.insert(
                "responseMimeType".to_owned(),
                Value::String("application/json".to_owned()),
            );
            if let Some(schema) = rf.get("json_schema").and_then(|js| js.get("schema")) {
                config.insert("responseSchema".to_owned(), schema.clone());
            }
        },
        _ => {},
    }
}

// -----------------------------------------------------------------------------
// Tool Definitions
// -----------------------------------------------------------------------------

/// Convert OpenAI `tools` array to Gemini `tools` with
/// `functionDeclarations`.
///
/// OpenAI wraps each tool in `{"type": "function", "function": {...}}`.
/// Gemini groups all functions under a single `tools` entry with
/// `functionDeclarations`.
fn convert_tools(obj: &Map<String, Value>) -> Option<Value> {
    let Value::Array(tools) = obj.get("tools")? else {
        return None;
    };

    let mut declarations = Vec::new();

    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }

        let Some(function) = tool.get("function") else {
            continue;
        };

        let mut decl = Map::new();

        if let Some(name) = function.get("name") {
            decl.insert("name".to_owned(), name.clone());
        }

        if let Some(desc) = function.get("description") {
            decl.insert("description".to_owned(), desc.clone());
        }

        if let Some(params) = function.get("parameters") {
            decl.insert("parameters".to_owned(), params.clone());
        }

        if !decl.is_empty() {
            declarations.push(Value::Object(decl));
        }
    }

    if declarations.is_empty() {
        return None;
    }

    Some(json!([{ "functionDeclarations": declarations }]))
}

// -----------------------------------------------------------------------------
// Tool Choice
// -----------------------------------------------------------------------------

/// Convert OpenAI `tool_choice` to Gemini `toolConfig`.
///
/// - `"auto"` → omitted (Gemini default, no need to send explicitly)
/// - `"required"` → `ANY` (must call a function)
/// - `"none"` → `NONE` (no function calls)
/// - `{"type": "function", "function": {"name": "..."}}` → `ANY` with `allowedFunctionNames`
fn convert_tool_choice(obj: &Map<String, Value>) -> Option<Value> {
    let tool_choice = obj.get("tool_choice")?;

    match tool_choice {
        Value::String(s) => match s.as_str() {
            "none" => Some(json!({ "functionCallingConfig": { "mode": "NONE" } })),
            "required" => Some(json!({ "functionCallingConfig": { "mode": "ANY" } })),
            // "auto" is the Gemini default — no need to send toolConfig.
            _ => None,
        },
        Value::Object(tc) => tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .map(|name| {
                json!({
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    }
                })
            }),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic request transformation
    // -------------------------------------------------------------------------

    #[test]
    fn basic_text_request() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[{"role":"user","content":"Hello"}]}"#;
        let result = transform_request(body).unwrap();

        assert_eq!(result.model, "gemini-1.5-pro");
        assert!(!result.stream);

        let parsed: Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(parsed["contents"][0]["role"], "user");
        assert_eq!(parsed["contents"][0]["parts"][0]["text"], "Hello");
        assert!(parsed.get("systemInstruction").is_none());
    }

    #[test]
    fn stream_flag_extracted() {
        let body = br#"{"model":"gemini-1.5-flash","stream":true,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        assert!(result.stream);
    }

    #[test]
    fn stream_string_true_rejected() {
        // "stream":"true" (string) must be rejected — the client clearly
        // intended SSE but would silently receive a JSON blob if we coerce
        // it to false. Return a 400 so the caller can fix the request.
        let body = br#"{"model":"gemini-1.5-flash","stream":"true","messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(err.contains("boolean"), "error should mention boolean: {err}");
    }

    #[test]
    fn stream_number_rejected() {
        let body = br#"{"model":"gemini-1.5-flash","stream":1,"messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(err.contains("boolean"), "error should mention boolean: {err}");
    }

    #[test]
    fn stream_null_becomes_false() {
        // null is explicitly allowed by the OpenAI spec and means false.
        let body = br#"{"model":"gemini-1.5-flash","stream":null,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        assert!(!result.stream);
    }

    #[test]
    fn missing_model_fails() {
        let body = br#"{"messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(err.contains("model"), "error should mention model: {err}");
    }

    #[test]
    fn non_json_body_fails() {
        let err = transform_request(b"not json").unwrap_err();
        assert!(err.contains("invalid JSON"), "error: {err}");
    }

    #[test]
    fn json_array_body_fails() {
        let err = transform_request(b"[1,2]").unwrap_err();
        assert!(err.contains("not a JSON object"), "error: {err}");
    }

    // -------------------------------------------------------------------------
    // Role mapping
    // -------------------------------------------------------------------------

    #[test]
    fn system_message_becomes_system_instruction() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"system","content":"Be helpful"},
            {"role":"user","content":"Hi"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["systemInstruction"]["parts"][0]["text"], "Be helpful",
            "system message should become systemInstruction"
        );
        assert_eq!(parsed["contents"].as_array().unwrap().len(), 1, "only user in contents");
        assert_eq!(parsed["contents"][0]["role"], "user");
    }

    #[test]
    fn developer_message_becomes_system_instruction() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"developer","content":"You are a coding assistant"},
            {"role":"user","content":"Hi"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["systemInstruction"]["parts"][0]["text"], "You are a coding assistant",
            "developer role should map to systemInstruction"
        );
    }

    #[test]
    fn multiple_system_messages_merged() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"system","content":"Part 1"},
            {"role":"developer","content":"Part 2"},
            {"role":"user","content":"Hi"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let parts = parsed["systemInstruction"]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "Part 1");
        assert_eq!(parts[1]["text"], "Part 2");
    }

    #[test]
    fn assistant_role_becomes_model() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"user","content":"Hello"},
            {"role":"assistant","content":"Hi there!"},
            {"role":"user","content":"How are you?"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["contents"][0]["role"], "user");
        assert_eq!(parsed["contents"][1]["role"], "model", "assistant → model");
        assert_eq!(parsed["contents"][1]["parts"][0]["text"], "Hi there!");
        assert_eq!(parsed["contents"][2]["role"], "user");
    }

    #[test]
    fn assistant_array_content_preserves_text_and_refusal_parts() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":[
                {"type":"text","text":"First"},
                {"type":"refusal","refusal":"Cannot do that"},
                {"type":"text","text":"Second"}
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["contents"][0]["role"], "model");
        assert_eq!(
            parsed["contents"][0]["parts"],
            json!([
                {"text": "First"},
                {"text": "Cannot do that"},
                {"text": "Second"}
            ])
        );
    }

    #[test]
    fn tool_result_with_name_field() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"tool","name":"get_weather","content":"{\"temp\":72}"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let part = &parsed["contents"][0]["parts"][0];
        assert_eq!(part["functionResponse"]["name"], "get_weather");
        assert_eq!(part["functionResponse"]["response"]["temp"], 72);
    }

    #[test]
    fn tool_result_resolved_via_tool_call_id() {
        // Standard OpenAI flow: assistant emits tool_calls with id,
        // then tool message references the id via tool_call_id.
        // No `name` field on the tool message.
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"user","content":"What is the weather?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}
            ]},
            {"role":"tool","tool_call_id":"call_abc123","content":"{\"temp\":72}"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        // The tool result should be in contents[2] (user, model, user+functionResponse)
        let tool_content = &parsed["contents"][2]["parts"][0];
        assert_eq!(
            tool_content["functionResponse"]["name"], "get_weather",
            "function name should be resolved from assistant tool_calls via tool_call_id"
        );
        assert_eq!(tool_content["functionResponse"]["response"]["temp"], 72);
    }

    #[test]
    fn colliding_tool_call_ids_resolve_to_nearest_preceding_name() {
        // A history-wide HashMap last-wins would bind both results to get_calendar.
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"user","content":"weather then calendar"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_vertex_0","type":"function","function":{"name":"get_weather","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_vertex_0","content":"{\"temp\":72}"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_vertex_0","type":"function","function":{"name":"get_calendar","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_vertex_0","content":"{\"event\":\"standup\"}"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let weather = &parsed["contents"][2]["parts"][0]["functionResponse"];
        let calendar = &parsed["contents"][4]["parts"][0]["functionResponse"];
        assert_eq!(weather["name"], "get_weather");
        assert_eq!(weather["response"]["temp"], 72);
        assert_eq!(calendar["name"], "get_calendar");
        assert_eq!(calendar["response"]["event"], "standup");
    }

    #[test]
    fn parallel_tool_results_grouped_into_one_user_turn() {
        // Google requires all parallel function responses from a single
        // assistant turn to be sent inside one `user` content block with
        // multiple `functionResponse` parts.  Emitting separate blocks
        // triggers a 400 from the Gemini API.
        let body = br#"{"model":"gemini-2.0-flash","messages":[
            {"role":"user","content":"What is the weather and time?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_w","type":"function","function":{"name":"get_weather","arguments":"{}"}},
                {"id":"call_t","type":"function","function":{"name":"get_time","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_w","content":"{\"temp\":72}"},
            {"role":"tool","tool_call_id":"call_t","content":"{\"time\":\"14:00\"}"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let contents = parsed["contents"].as_array().unwrap();
        // user(1) + model(1) + user-with-both-responses(1) = 3 total
        assert_eq!(
            contents.len(),
            3,
            "parallel tool results must be merged into a single user turn, got: {contents:?}"
        );

        let tool_turn = &contents[2];
        assert_eq!(tool_turn["role"], "user");
        let parts = tool_turn["parts"].as_array().unwrap();
        assert_eq!(
            parts.len(),
            2,
            "both functionResponse parts must live in the same user content block"
        );
        assert_eq!(parts[0]["functionResponse"]["name"], "get_weather");
        assert_eq!(parts[0]["functionResponse"]["response"]["temp"], 72);
        assert_eq!(parts[1]["functionResponse"]["name"], "get_time");
        assert_eq!(parts[1]["functionResponse"]["response"]["time"], "14:00");
    }

    #[test]
    fn sequential_tool_rounds_are_not_merged() {
        // Two separate assistant→tool round-trips must NOT be merged: each
        // set of responses belongs to a different assistant turn.
        let body = br#"{"model":"gemini-2.0-flash","messages":[
            {"role":"user","content":"first"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_a","type":"function","function":{"name":"fn_a","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_a","content":"{\"a\":1}"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_b","type":"function","function":{"name":"fn_b","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"call_b","content":"{\"b\":2}"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let contents = parsed["contents"].as_array().unwrap();
        // user + model + user(tool_a) + model + user(tool_b) = 5
        assert_eq!(
            contents.len(),
            5,
            "sequential tool rounds must produce separate user turns, got: {contents:?}"
        );
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["name"], "fn_a");
        assert_eq!(contents[4]["parts"][0]["functionResponse"]["name"], "fn_b");
    }

    #[test]
    fn tool_result_json_non_object_content_wrapped() {
        // Valid non-object JSON (array, number, etc.) must be wrapped in
        // {"result": ...} — Gemini functionResponse.response requires a Struct.
        for (content, label) in [("[1,2,3]", "array"), ("42", "number"), ("true", "boolean")] {
            let body = format!(
                r#"{{"model":"gemini-1.5-pro","messages":[{{"role":"tool","name":"f","content":"{content}"}}]}}"#
            );
            let result = transform_request(body.as_bytes()).unwrap();
            let parsed: Value = serde_json::from_slice(&result.body).unwrap();
            let resp = &parsed["contents"][0]["parts"][0]["functionResponse"]["response"];
            assert!(
                resp.is_object(),
                "{label} tool content must produce an object response, got: {resp}"
            );
            assert_eq!(
                resp["result"].as_str().unwrap_or(""),
                content,
                "{label} content should be preserved as the 'result' string"
            );
        }
    }

    #[test]
    fn tool_result_unresolvable_falls_back_to_unknown() {
        // No name, no matching tool_call_id in history → "unknown"
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"tool","tool_call_id":"call_orphan","content":"data"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["contents"][0]["parts"][0]["functionResponse"]["name"], "unknown",
            "unresolvable tool_call_id should fall back to unknown"
        );
    }

    #[test]
    fn tool_result_non_json_content_wrapped() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"tool","name":"search","content":"no results found"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["contents"][0]["parts"][0]["functionResponse"]["response"]["result"], "no results found",
            "non-JSON tool content should be wrapped in {{\"result\": ...}}"
        );
    }

    #[test]
    fn tool_result_array_content_preserves_text_parts() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"tool","name":"search","content":[
                {"type":"text","text":"first result"},
                {"type":"text","text":"second result"}
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["contents"][0]["parts"][0]["functionResponse"]["response"]["result"],
            "first result\nsecond result"
        );
    }

    #[test]
    fn tool_result_array_content_preserves_json_object() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"tool","name":"get_weather","content":[
                {"type":"text","text":"{\"temp\":72}"}
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["contents"][0]["parts"][0]["functionResponse"]["response"]["temp"],
            72
        );
    }

    #[test]
    fn unknown_role_dropped() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"custom_role","content":"ignored"},
            {"role":"user","content":"Hi"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["contents"].as_array().unwrap().len(),
            1,
            "unknown role should be dropped"
        );
    }

    #[test]
    fn message_without_role_skipped() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[{"content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(parsed["contents"].as_array().unwrap().is_empty());
    }

    // -------------------------------------------------------------------------
    // Assistant tool calls
    // -------------------------------------------------------------------------

    #[test]
    fn assistant_tool_calls_become_function_call_parts() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let part = &parsed["contents"][0]["parts"][0];
        assert_eq!(parsed["contents"][0]["role"], "model");
        assert_eq!(part["functionCall"]["name"], "get_weather");
        assert_eq!(part["functionCall"]["args"]["city"], "NYC");
        assert!(
            part["functionCall"].get("id").is_none(),
            "Vertex native REST rejects functionCall.id; OpenAI tool_calls[].id stays client-side"
        );
    }

    #[test]
    fn assistant_tool_call_thought_signature_round_trips() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"},
                    "extra_content":{"google":{"thought_signature":"sig-abc"}}
                }
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();
        let part = &parsed["contents"][0]["parts"][0];

        assert_eq!(part["thoughtSignature"], "sig-abc");
        assert_eq!(part["functionCall"]["name"], "get_weather");
        assert!(
            part["functionCall"].get("thoughtSignature").is_none(),
            "signature belongs on the Part, not inside functionCall"
        );
        assert!(
            part["functionCall"].get("id").is_none(),
            "do not echo OpenAI/Google call ids on Vertex native functionCall"
        );
    }

    #[test]
    fn assistant_tool_call_google_id_not_sent_on_function_call() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"adk-call-123","type":"function","function":{"name":"get_weather","arguments":"{}"}}
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();
        let part = &parsed["contents"][0]["parts"][0];

        assert_eq!(part["functionCall"]["name"], "get_weather");
        assert!(part["functionCall"].get("id").is_none());
    }

    #[test]
    fn tool_result_does_not_send_function_response_id() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"adk-call-123","type":"function","function":{"name":"get_weather","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"adk-call-123","content":"{\"temp\":72}"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();
        let fr = &parsed["contents"][1]["parts"][0]["functionResponse"];

        assert_eq!(fr["name"], "get_weather");
        assert!(fr.get("id").is_none());
    }

    #[test]
    fn assistant_tool_call_invalid_json_args_rejected() {
        // arguments is not valid JSON — must be rejected with a clear error.
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"not-json"}}
            ]}
        ]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(
            err.contains("not valid JSON"),
            "error should mention invalid JSON: {err}"
        );
    }

    #[test]
    fn assistant_tool_call_non_object_args_rejected() {
        // arguments is valid JSON but not an object — must be rejected.
        for bad in [r#""[1,2,3]""#, r#""42""#, r#""true""#, r#""\"a string\"""#] {
            let body = format!(
                r#"{{"model":"gemini-1.5-pro","messages":[{{"role":"assistant","content":null,"tool_calls":[{{"id":"c","type":"function","function":{{"name":"f","arguments":{bad}}}}}]}}]}}"#
            );
            let err = transform_request(body.as_bytes()).unwrap_err();
            assert!(
                err.contains("JSON object"),
                "expected object type error for arguments={bad}, got: {err}"
            );
        }
    }

    #[test]
    fn assistant_tool_call_missing_arguments_rejected() {
        // function.arguments is absent entirely — must be rejected, not defaulted to {}.
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"get_weather"}}
            ]}
        ]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(
            err.contains("function.arguments"),
            "error should mention function.arguments for missing field: {err}"
        );
    }

    #[test]
    fn assistant_text_and_tool_calls_combined() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"assistant","content":"Let me check","tool_calls":[
                {"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}
            ]}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let parts = parsed["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2, "text + functionCall");
        assert_eq!(parts[0]["text"], "Let me check");
        assert_eq!(parts[1]["functionCall"]["name"], "search");
    }

    // -------------------------------------------------------------------------
    // Multipart content (images)
    // -------------------------------------------------------------------------

    #[test]
    fn base64_image_becomes_inline_data() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,abc123"}},
            {"type":"text","text":"What is this?"}
        ]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let parts = parsed["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[0]["inlineData"]["data"], "abc123");
        assert_eq!(parts[1]["text"], "What is this?");
    }

    #[test]
    fn url_image_becomes_file_data() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"https://example.com/photo.png"}}
        ]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["contents"][0]["parts"][0]["fileData"]["fileUri"],
            "https://example.com/photo.png"
        );
        assert_eq!(parsed["contents"][0]["parts"][0]["fileData"]["mimeType"], "image/png");
    }

    // -------------------------------------------------------------------------
    // Generation config
    // -------------------------------------------------------------------------

    #[test]
    fn parameters_nested_in_generation_config() {
        let body = br#"{"model":"gemini-1.5-pro","temperature":0.7,"top_p":0.9,"max_tokens":1024,"stop":["\n\n"],"seed":42,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let gc = &parsed["generationConfig"];
        assert_eq!(gc["temperature"], 0.7);
        assert_eq!(gc["topP"], 0.9);
        assert_eq!(gc["maxOutputTokens"], 1024);
        assert_eq!(gc["stopSequences"][0], "\n\n");
        assert_eq!(gc["seed"], 42);
    }

    #[test]
    fn max_completion_tokens_takes_precedence() {
        let body = br#"{"model":"gemini-1.5-pro","max_tokens":500,"max_completion_tokens":1000,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(
            parsed["generationConfig"]["maxOutputTokens"], 1000,
            "max_completion_tokens should take precedence"
        );
    }

    #[test]
    fn no_parameters_omits_generation_config() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(
            parsed.get("generationConfig").is_none(),
            "empty config should be omitted"
        );
    }

    #[test]
    fn response_format_json_object() {
        let body = br#"{"model":"gemini-1.5-pro","response_format":{"type":"json_object"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["generationConfig"]["responseMimeType"], "application/json");
    }

    #[test]
    fn response_format_json_schema() {
        let body = br#"{"model":"gemini-1.5-pro","response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"name":{"type":"string"}}}}},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["generationConfig"]["responseMimeType"], "application/json");
        assert_eq!(parsed["generationConfig"]["responseSchema"]["type"], "object");
    }

    // -------------------------------------------------------------------------
    // Tool definitions
    // -------------------------------------------------------------------------

    #[test]
    fn tools_converted_to_function_declarations() {
        let body = br#"{"model":"gemini-1.5-pro","tools":[
            {"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}
        ],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let decl = &parsed["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "get_weather");
        assert_eq!(decl["description"], "Get weather");
        assert_eq!(decl["parameters"]["type"], "object");
    }

    #[test]
    fn non_function_tools_skipped() {
        let body = br#"{"model":"gemini-1.5-pro","tools":[
            {"type":"code_interpreter"},
            {"type":"function","function":{"name":"search","description":"Search"}}
        ],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let decls = parsed["tools"][0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1, "only function tools should be converted");
        assert_eq!(decls[0]["name"], "search");
    }

    // -------------------------------------------------------------------------
    // Tool choice
    // -------------------------------------------------------------------------

    #[test]
    fn tool_choice_auto_omits_tool_config() {
        let body = br#"{"model":"gemini-1.5-pro","tool_choice":"auto","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(
            parsed.get("toolConfig").is_none(),
            "auto is the Gemini default — toolConfig should be omitted"
        );
    }

    #[test]
    fn tool_choice_none() {
        let body = br#"{"model":"gemini-1.5-pro","tool_choice":"none","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["toolConfig"]["functionCallingConfig"]["mode"], "NONE");
    }

    #[test]
    fn tool_choice_required() {
        let body =
            br#"{"model":"gemini-1.5-pro","tool_choice":"required","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    }

    #[test]
    fn tool_choice_specific_function() {
        let body = br#"{"model":"gemini-1.5-pro","tool_choice":{"type":"function","function":{"name":"get_weather"}},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            parsed["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "get_weather"
        );
    }

    // -------------------------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn empty_system_content_omitted() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"system","content":""},
            {"role":"user","content":"Hi"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(
            parsed.get("systemInstruction").is_none(),
            "empty system content should be omitted"
        );
    }

    #[test]
    fn model_with_path_traversal_rejected() {
        let body = br#"{"model":"../../other-model","messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(err.contains("alphanumeric"), "{err}");
    }

    #[test]
    fn empty_model_rejected() {
        let body = br#"{"model":"","messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(err.contains("empty"), "error should mention empty: {err}");
    }

    #[test]
    fn whitespace_only_model_rejected() {
        // "   ".is_empty() is false — trimming must happen before the empty check.
        let body = br#"{"model":"   ","messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(err.contains("empty"), "whitespace-only model must be rejected: {err}");
    }

    #[test]
    fn whitespace_padded_model_is_trimmed() {
        let body = br#"{"model":"  gemini-2.0-flash  ","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        assert_eq!(
            result.model, "gemini-2.0-flash",
            "whitespace must be stripped from model"
        );
    }

    #[test]
    fn model_with_url_unsafe_chars_rejected() {
        for bad in ["gemini@2.0", "model[1]", "my%model", "my model", "gemini:2.0"] {
            let body = format!(r#"{{"model":"{bad}","messages":[{{"role":"user","content":"Hi"}}]}}"#);
            let err = transform_request(body.as_bytes()).unwrap_err();
            assert!(err.contains("alphanumeric"), "'{bad}' should be rejected, got: {err}");
        }
    }

    #[test]
    fn empty_messages_array() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(parsed["contents"].as_array().unwrap().is_empty());
    }

    #[test]
    fn system_multipart_text_blocks() {
        let body = br#"{"model":"gemini-1.5-pro","messages":[
            {"role":"system","content":[{"type":"text","text":"Rule 1"},{"type":"text","text":"Rule 2"}]},
            {"role":"user","content":"Hi"}
        ]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let parts = parsed["systemInstruction"]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "Rule 1");
        assert_eq!(parts[1]["text"], "Rule 2");
    }

    #[test]
    fn stop_string_wrapped_in_array() {
        let body = br#"{"model":"gemini-1.5-pro","stop":"\n","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        let stops = parsed["generationConfig"]["stopSequences"].as_array().unwrap();
        assert_eq!(stops.len(), 1);
        assert_eq!(
            stops[0], "\n",
            "string stop should be wrapped in a single-element array"
        );
    }

    #[test]
    fn penalty_parameters_mapped() {
        let body = br#"{"model":"gemini-1.5-pro","presence_penalty":0.5,"frequency_penalty":0.3,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["generationConfig"]["presencePenalty"], 0.5);
        assert_eq!(parsed["generationConfig"]["frequencyPenalty"], 0.3);
    }

    #[test]
    fn streaming_include_usage_is_extracted_not_forwarded() {
        let body = br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert!(result.include_usage);
        assert!(result.stream);
        assert!(
            parsed.get("stream_options").is_none(),
            "Gemini has no stream_options; must not be forwarded"
        );
    }

    #[test]
    fn include_usage_false_is_accepted() {
        let body = br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        assert!(!result.include_usage);
    }

    #[test]
    fn include_usage_null_is_accepted() {
        let body = br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":null},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        assert!(!result.include_usage);
    }

    #[test]
    fn include_usage_string_true_rejected() {
        // Non-boolean must be rejected with a clear error, not silently ignored.
        let body = br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":"true"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(
            err.contains("stream_options.include_usage") && err.contains("boolean"),
            "expected boolean type error, got: {err}"
        );
    }

    #[test]
    fn include_usage_number_rejected() {
        let body = br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":{"include_usage":1},"messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(
            err.contains("stream_options.include_usage") && err.contains("boolean"),
            "expected boolean type error, got: {err}"
        );
    }

    #[test]
    fn include_usage_without_stream_is_rejected() {
        let body = br#"{"model":"gemini-2.0-flash","stream":false,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(
            err.contains("stream_options") && err.contains("stream"),
            "stream_options without streaming must be rejected: {err}"
        );
    }

    #[test]
    fn non_object_stream_options_is_rejected() {
        let body = br#"{"model":"gemini-2.0-flash","stream":true,"stream_options":"invalid","messages":[{"role":"user","content":"Hi"}]}"#;
        let err = transform_request(body).unwrap_err();
        assert!(
            err.contains("stream_options") && err.contains("object"),
            "invalid stream_options container must be rejected: {err}"
        );
    }

    #[test]
    fn n_mapped_to_candidate_count() {
        let body = br#"{"model":"gemini-2.0-flash","n":3,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["generationConfig"]["candidateCount"], 3);
        assert_eq!(result.candidate_count, 3);
    }

    #[test]
    fn logprobs_mapped() {
        let body = br#"{"model":"gemini-2.0-flash","logprobs":true,"top_logprobs":5,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["generationConfig"]["responseLogprobs"], true);
        assert_eq!(parsed["generationConfig"]["logprobs"], 5);
    }
}
