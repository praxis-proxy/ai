// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Chat Completions to Vertex AI Gemini request transformation.
//!
//! Translates the full request body: messages → contents, role mapping,
//! parameter nesting under `generationConfig`, tool definitions, and
//! `stream` flag extraction for URL path selection.

use std::collections::HashMap;

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

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "request body must contain a \"model\" field".to_owned())?
        .to_owned();

    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let mut gemini = Map::new();

    // Messages → contents + systemInstruction
    let (contents, system_instruction) = convert_messages(obj);
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

    Ok(TransformResult { body, model, stream })
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
fn convert_messages(obj: &Map<String, Value>) -> (Vec<Value>, Option<Value>) {
    let Some(Value::Array(messages)) = obj.get("messages") else {
        return (Vec::new(), None);
    };

    // Resolve tool_call_id → function name for tool result messages.
    let tool_name_map = build_tool_call_name_map(messages);

    let mut contents = Vec::new();
    let mut system_parts = Vec::new();

    for msg in messages {
        let Some(role) = msg.get("role").and_then(Value::as_str) else {
            continue;
        };

        match role {
            "system" | "developer" => collect_system_parts(&mut system_parts, msg),
            "user" => convert_user_message(&mut contents, msg),
            "assistant" => convert_assistant_message(&mut contents, msg),
            "tool" | "function" => convert_tool_result(&mut contents, msg, &tool_name_map),
            _ => {
                warn!(role, "dropping message with unknown role");
            },
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(json!({ "parts": system_parts }))
    };

    (contents, system_instruction)
}

/// Build a map from `tool_call_id` → function name by scanning assistant
/// messages that contain `tool_calls`.
///
/// This is needed because OpenAI `tool` messages only carry `tool_call_id`
/// (not `name`), but Gemini `functionResponse` requires the function name.
fn build_tool_call_name_map(messages: &[Value]) -> HashMap<String, String> {
    let mut map = HashMap::new();

    for msg in messages {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }

        let Some(Value::Array(tool_calls)) = msg.get("tool_calls") else {
            continue;
        };

        for tc in tool_calls {
            let id = tc.get("id").and_then(Value::as_str);
            let name = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str);

            if let (Some(id), Some(name)) = (id, name) {
                map.insert(id.to_owned(), name.to_owned());
            }
        }
    }

    map
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
/// Handles plain text responses and tool calls (`functionCall` parts).
fn convert_assistant_message(contents: &mut Vec<Value>, msg: &Value) {
    let mut parts = Vec::new();

    if let Some(text) = msg.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        parts.push(json!({ "text": text }));
    }

    if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
        for tc in tool_calls {
            if let Some(fc) = convert_tool_call(tc) {
                parts.push(fc);
            }
        }
    }

    if !parts.is_empty() {
        contents.push(json!({ "role": "model", "parts": parts }));
    }
}

/// Convert a single OpenAI tool call to a Gemini `functionCall` part.
fn convert_tool_call(tc: &Value) -> Option<Value> {
    let name = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str)?;

    let args_str = tc
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or("{}");

    let args: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));

    Some(json!({
        "functionCall": {
            "name": name,
            "args": args,
        }
    }))
}

// -----------------------------------------------------------------------------
// Tool Result Messages
// -----------------------------------------------------------------------------

/// Convert a `tool` or `function` role message to a Gemini
/// `functionResponse` part.
///
/// Resolves the function name in order of preference:
/// 1. `msg["name"]` — present on deprecated `function` messages and some clients that add it to `tool` messages for
///    convenience
/// 2. `tool_call_id` → lookup in `tool_name_map` built from prior assistant `tool_calls`
/// 3. Fallback to `"unknown"` with a warning (Gemini will likely reject this with 400)
fn convert_tool_result(contents: &mut Vec<Value>, msg: &Value, tool_name_map: &HashMap<String, String>) {
    let name = resolve_tool_function_name(msg, tool_name_map);

    let content = msg.get("content").and_then(Value::as_str).unwrap_or("");

    let response: Value = serde_json::from_str(content).unwrap_or_else(|_| json!({ "result": content }));

    contents.push(json!({
        "role": "user",
        "parts": [{
            "functionResponse": {
                "name": name,
                "response": response,
            }
        }]
    }));
}

/// Resolve the function name for a tool/function result message.
///
/// Tries `msg["name"]` first (always present on deprecated `function`
/// messages, sometimes present on `tool` messages). Falls back to
/// looking up `tool_call_id` in the map built from assistant
/// `tool_calls`. Logs a warning if neither source provides a name.
fn resolve_tool_function_name<'a>(msg: &'a Value, tool_name_map: &'a HashMap<String, String>) -> &'a str {
    // 1. Direct name field (deprecated `function` messages, some clients)
    if let Some(name) = msg.get("name").and_then(Value::as_str)
        && !name.is_empty()
    {
        return name;
    }

    // 2. Lookup via tool_call_id → assistant tool_calls
    if let Some(tool_call_id) = msg.get("tool_call_id").and_then(Value::as_str) {
        if let Some(name) = tool_name_map.get(tool_call_id) {
            return name.as_str();
        }
        warn!(
            tool_call_id,
            "tool message tool_call_id has no matching assistant tool_call; \
             using \"unknown\" as function name"
        );
    } else {
        warn!("tool message has neither name nor tool_call_id");
    }

    "unknown"
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
    fn n_mapped_to_candidate_count() {
        let body = br#"{"model":"gemini-2.0-flash","n":3,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed["generationConfig"]["candidateCount"], 3);
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
