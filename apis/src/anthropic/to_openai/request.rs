// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages to Chat Completions-compatible request transformation.

use serde_json::{Map, Value, json};
use tracing::warn;

// -----------------------------------------------------------------------------
// Request Transformation
// -----------------------------------------------------------------------------

/// Transform an Anthropic Messages request body into Chat
/// Completions-compatible format.
///
/// Returns the transformed JSON bytes, or an error message.
pub(crate) fn transform_request(body: &[u8]) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;

    let Some(obj) = value.as_object() else {
        return Err("request body is not a JSON object".to_owned());
    };

    let mut chat = Map::new();

    if let Some(model) = obj.get("model") {
        chat.insert("model".to_owned(), model.clone());
    }

    let mut messages = Vec::new();
    hoist_system(&mut messages, obj);
    convert_messages(&mut messages, obj);
    chat.insert("messages".to_owned(), Value::Array(messages));

    if let Some(max_tokens) = obj.get("max_tokens") {
        chat.insert("max_completion_tokens".to_owned(), max_tokens.clone());
    }

    if let Some(stream) = obj.get("stream") {
        chat.insert("stream".to_owned(), stream.clone());
    }

    map_parameters(&mut chat, obj);
    convert_tools(&mut chat, obj);
    convert_parallel_tool_calls(&mut chat, obj);
    convert_tool_choice(&mut chat, obj);

    serde_json::to_vec(&Value::Object(chat)).map_err(|e| format!("serialization failed: {e}"))
}

// -----------------------------------------------------------------------------
// System Message Hoisting
// -----------------------------------------------------------------------------

/// Hoist Anthropic top-level `system` to a Chat Completions system message.
fn hoist_system(messages: &mut Vec<Value>, obj: &Map<String, Value>) {
    let Some(system) = obj.get("system") else {
        return;
    };

    let content = match system {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(text.to_owned());
                }
            }
            parts.join("\n")
        },
        _ => return,
    };

    if !content.is_empty() {
        messages.push(json!({"role": "system", "content": content}));
    }
}

// -----------------------------------------------------------------------------
// Message Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic messages array to Chat Completions messages.
fn convert_messages(messages: &mut Vec<Value>, obj: &Map<String, Value>) {
    let Some(Value::Array(anthropic_messages)) = obj.get("messages") else {
        return;
    };

    for msg in anthropic_messages {
        let Some(role) = msg.get("role").and_then(Value::as_str) else {
            continue;
        };

        match msg.get("content") {
            Some(Value::String(text)) => {
                messages.push(json!({"role": role, "content": text}));
            },
            Some(Value::Array(blocks)) => {
                convert_content_blocks(messages, role, blocks);
            },
            _ => {
                messages.push(json!({"role": role, "content": ""}));
            },
        }
    }
}

/// Convert typed content blocks to Chat Completions-compatible format.
fn convert_content_blocks(messages: &mut Vec<Value>, role: &str, blocks: &[Value]) {
    let mut text_parts = Vec::new();
    let mut content_parts: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
        convert_single_block(
            block,
            block_type,
            messages,
            role,
            &mut text_parts,
            &mut content_parts,
            &mut tool_calls,
        );
    }

    finalize_content_blocks(messages, role, &mut text_parts, &mut content_parts, tool_calls);
}

/// Process a single content block within a message.
#[expect(
    clippy::too_many_arguments,
    reason = "accumulator pattern requires passing all state"
)]
fn convert_single_block(
    block: &Value,
    block_type: &str,
    messages: &mut Vec<Value>,
    role: &str,
    text_parts: &mut Vec<String>,
    content_parts: &mut Vec<Value>,
    tool_calls: &mut Vec<Value>,
) {
    match block_type {
        "text" => convert_text_block(block, text_parts, content_parts),
        "image" => convert_image_block(block, content_parts),
        "search_result" => convert_search_result_block(block, text_parts, content_parts),
        "document" => convert_document_block(block, text_parts, content_parts),
        "tool_use" => convert_tool_use_block(block, tool_calls),
        "tool_result" => {
            flush_text_parts(messages, text_parts, content_parts, role);
            convert_tool_result_block(block, messages);
        },
        "thinking" | "redacted_thinking" => {
            warn!(block_type, "dropping unsupported Anthropic content block");
        },
        _ => {
            warn!(block_type, "dropping unknown Anthropic content block type");
        },
    }
}

/// Convert a text content block.
fn convert_text_block(block: &Value, text_parts: &mut Vec<String>, content_parts: &mut Vec<Value>) {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        append_text_content(text, text_parts, content_parts);
    }
}

/// Convert an image content block.
fn convert_image_block(block: &Value, content_parts: &mut Vec<Value>) {
    if let Some(source) = block.get("source")
        && let Some(url_val) = convert_image_source(source)
    {
        content_parts.push(json!({"type": "image_url", "image_url": {"url": url_val}}));
    }
}

/// Convert a `search_result` block to backend-visible text context.
fn convert_search_result_block(block: &Value, text_parts: &mut Vec<String>, content_parts: &mut Vec<Value>) {
    if let Some(text) = flatten_search_result(block) {
        append_text_content(&text, text_parts, content_parts);
    }
}

/// Convert a `document` block to backend-visible text context.
fn convert_document_block(block: &Value, text_parts: &mut Vec<String>, content_parts: &mut Vec<Value>) {
    if let Some(text) = flatten_document(block) {
        append_text_content(&text, text_parts, content_parts);
    }
}

/// Append one Chat Completions text content part and its string equivalent.
fn append_text_content(text: &str, text_parts: &mut Vec<String>, content_parts: &mut Vec<Value>) {
    text_parts.push(text.to_owned());
    content_parts.push(json!({"type": "text", "text": text}));
}

/// Convert a `tool_use` content block to a Chat Completions tool call.
fn convert_tool_use_block(block: &Value, tool_calls: &mut Vec<Value>) {
    let id = block.get("id").and_then(Value::as_str).unwrap_or("");
    let name = block.get("name").and_then(Value::as_str).unwrap_or("");
    let input = block.get("input").cloned().unwrap_or_else(|| Value::Object(Map::new()));
    let args = serde_json::to_string(&input).unwrap_or_default();

    tool_calls.push(json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": args}
    }));
}

/// Convert a `tool_result` content block to a Chat Completions tool message.
fn convert_tool_result_block(block: &Value, messages: &mut Vec<Value>) {
    let tool_call_id = block.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
    let mut result_content = extract_tool_result_content(block);
    let image_content = extract_tool_result_image_content(block);

    if block.get("is_error").and_then(Value::as_bool) == Some(true) {
        result_content = mark_tool_result_error(result_content);
    }

    messages.push(json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result_content
    }));

    if !image_content.is_empty() {
        messages.push(json!({
            "role": "user",
            "content": image_content
        }));
    }
}

/// Emit the final message for accumulated content and tool calls.
fn finalize_content_blocks(
    messages: &mut Vec<Value>,
    role: &str,
    text_parts: &mut Vec<String>,
    content_parts: &mut Vec<Value>,
    tool_calls: Vec<Value>,
) {
    if role == "assistant" && !tool_calls.is_empty() {
        let mut msg = json!({"role": "assistant"});
        if let Some(obj) = msg.as_object_mut() {
            if !text_parts.is_empty() {
                obj.insert("content".to_owned(), Value::String(text_parts.join("")));
            }
            obj.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        }
        messages.push(msg);
    } else {
        flush_text_parts(messages, text_parts, content_parts, role);
    }
}

/// Flush accumulated text/content parts as a message.
fn flush_text_parts(
    messages: &mut Vec<Value>,
    text_parts: &mut Vec<String>,
    content_parts: &mut Vec<Value>,
    role: &str,
) {
    if content_parts.is_empty() && text_parts.is_empty() {
        return;
    }

    if content_parts.len() == 1
        && content_parts
            .first()
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str)
            == Some("text")
    {
        messages.push(json!({"role": role, "content": text_parts.join("")}));
    } else if !content_parts.is_empty() {
        messages.push(json!({"role": role, "content": std::mem::take(content_parts)}));
    }

    text_parts.clear();
    content_parts.clear();
}

// -----------------------------------------------------------------------------
// Image Source Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic image source to an `image_url` URL string.
fn convert_image_source(source: &Value) -> Option<String> {
    let source_type = source.get("type").and_then(Value::as_str)?;

    match source_type {
        "base64" => {
            let media_type = source.get("media_type").and_then(Value::as_str)?;
            let data = source.get("data").and_then(Value::as_str)?;
            Some(format!("data:{media_type};base64,{data}"))
        },
        "url" => source.get("url").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Tool Result Content Extraction
// -----------------------------------------------------------------------------

/// Extract text content from a `tool_result` block.
fn extract_tool_result_content(block: &Value) -> String {
    match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut text_parts = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            text_parts.push(text.to_owned());
                        }
                    },
                    Some("search_result") => {
                        if let Some(text) = flatten_search_result(part) {
                            text_parts.push(text);
                        }
                    },
                    Some("document") => {
                        if let Some(text) = flatten_document(part) {
                            text_parts.push(text);
                        }
                    },
                    _ => {},
                }
            }
            text_parts.join("\n")
        },
        _ => String::new(),
    }
}

/// Preserve Anthropic's `tool_result.is_error` semantic in text-only tool messages.
fn mark_tool_result_error(mut content: String) -> String {
    if content.is_empty() {
        "Anthropic tool_result error".to_owned()
    } else {
        content.insert_str(0, "Anthropic tool_result error:\n");
        content
    }
}

/// Extract image content from a `tool_result` block.
fn extract_tool_result_image_content(block: &Value) -> Vec<Value> {
    let Some(Value::Array(parts)) = block.get("content") else {
        return Vec::new();
    };

    let mut image_parts = Vec::new();
    for part in parts {
        if part.get("type").and_then(Value::as_str) == Some("image") {
            convert_image_block(part, &mut image_parts);
        }
    }
    image_parts
}

/// Flatten an Anthropic `search_result` block to plain text.
fn flatten_search_result(block: &Value) -> Option<String> {
    let title = block
        .get("title")
        .and_then(Value::as_str)
        .filter(|title| !title.is_empty());
    let source = block
        .get("source")
        .and_then(Value::as_str)
        .filter(|source| !source.is_empty());
    let content = extract_text_blocks(block.get("content"));

    if title.is_none() && source.is_none() && content.is_empty() {
        return None;
    }

    let mut lines = Vec::new();

    if let Some(title) = title {
        lines.push(format!("Search result: {}", quote_label_value(title)));
    } else {
        lines.push("Search result".to_owned());
    }

    if let Some(source) = source {
        lines.push(format!("Source: {}", quote_label_value(source)));
    }

    if !content.is_empty() {
        lines.push("Content:".to_owned());
    }

    lines.extend(content);
    non_empty_lines(&lines)
}

/// Flatten an Anthropic `document` block to plain text.
fn flatten_document(block: &Value) -> Option<String> {
    let title = block
        .get("title")
        .and_then(Value::as_str)
        .filter(|title| !title.is_empty());
    let context = block
        .get("context")
        .and_then(Value::as_str)
        .filter(|context| !context.is_empty());
    let source_text = flatten_document_source(block.get("source"));

    if title.is_none() && context.is_none() && source_text.is_none() {
        return None;
    }

    let mut lines = Vec::new();

    if let Some(title) = title {
        lines.push(format!("Document: {}", quote_label_value(title)));
    } else {
        lines.push("Document".to_owned());
    }

    if let Some(context) = context {
        lines.push(format!("Context: {}", quote_label_value(context)));
    }

    if let Some(source_text) = source_text {
        lines.push(source_text);
    }

    non_empty_lines(&lines)
}

/// Flatten a `document.source` value to extractable text or a stable reference.
fn flatten_document_source(source: Option<&Value>) -> Option<String> {
    let source = source?;
    let source_type = source.get("type").and_then(Value::as_str)?;

    match source_type {
        "text" => source
            .get("data")
            .and_then(Value::as_str)
            .filter(|data| !data.is_empty())
            .map(|data| format!("Content:\n{data}")),
        "content" => {
            let lines = extract_text_blocks(source.get("content"));
            non_empty_lines(&lines).map(|content| format!("Content:\n{content}"))
        },
        "url" => source
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
            .map(|url| format!("Source: {}", quote_label_value(url))),
        "file" => source
            .get("file_id")
            .and_then(Value::as_str)
            .filter(|file_id| !file_id.is_empty())
            .map(|file_id| format!("Source: {}", quote_label_value(&format!("file:{file_id}")))),
        "base64" => source
            .get("media_type")
            .and_then(Value::as_str)
            .filter(|media_type| !media_type.is_empty())
            .map(|media_type| format!("Source: {}", quote_label_value(&format!("base64:{media_type}")))),
        _ => None,
    }
}

/// Quote metadata values so embedded newlines cannot forge flattening labels.
fn quote_label_value(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

/// Extract text from an array of Anthropic text blocks.
fn extract_text_blocks(value: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(blocks)) = value else {
        return Vec::new();
    };

    blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Join lines if at least one line contains content.
fn non_empty_lines(lines: &[String]) -> Option<String> {
    lines.iter().any(|line| !line.is_empty()).then(|| lines.join("\n"))
}

// -----------------------------------------------------------------------------
// Parameter Mapping
// -----------------------------------------------------------------------------

/// Map Anthropic parameters to Chat Completions-compatible equivalents.
///
/// `top_k` has no standard Chat Completions equivalent but is preserved
/// as an extra body parameter for backends that support it
/// (e.g. vLLM).
fn map_parameters(chat: &mut Map<String, Value>, obj: &Map<String, Value>) {
    if let Some(stop) = obj.get("stop_sequences") {
        chat.insert("stop".to_owned(), stop.clone());
    }

    if let Some(temp) = obj.get("temperature") {
        chat.insert("temperature".to_owned(), temp.clone());
    }

    if let Some(top_p) = obj.get("top_p") {
        chat.insert("top_p".to_owned(), top_p.clone());
    }

    if let Some(top_k) = obj.get("top_k") {
        chat.insert("top_k".to_owned(), top_k.clone());
    }
}

// -----------------------------------------------------------------------------
// Tool Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic tool definitions to Chat Completions function tools.
fn convert_tools(chat: &mut Map<String, Value>, obj: &Map<String, Value>) {
    let Some(Value::Array(tools)) = obj.get("tools") else {
        return;
    };

    let mut chat_tools = Vec::new();

    for tool in tools {
        if let Some(chat_tool) = convert_tool_definition(tool) {
            chat_tools.push(chat_tool);
        }
    }

    if !chat_tools.is_empty() {
        chat.insert("tools".to_owned(), Value::Array(chat_tools));
    }
}

/// Convert one Anthropic client tool definition to a Chat Completions tool.
fn convert_tool_definition(tool: &Value) -> Option<Value> {
    let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or("custom");

    if tool_type.starts_with("web_search") || tool_type.starts_with("bash") || tool_type.starts_with("text_editor") {
        warn!(tool_type, "dropping server-side Anthropic tool");
        return None;
    }

    let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
    let description = tool.get("description").and_then(Value::as_str).unwrap_or("");
    let parameters = tool
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object"}));

    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name.to_owned()));
    function.insert("description".to_owned(), Value::String(description.to_owned()));
    function.insert("parameters".to_owned(), parameters);
    if let Some(strict) = tool.get("strict").and_then(Value::as_bool) {
        function.insert("strict".to_owned(), Value::Bool(strict));
    }

    Some(json!({
        "type": "function",
        "function": function
    }))
}

// -----------------------------------------------------------------------------
// Tool Choice Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic `disable_parallel_tool_use` to Chat Completions format.
fn convert_parallel_tool_calls(chat: &mut Map<String, Value>, obj: &Map<String, Value>) {
    let Some(Value::Object(tool_choice)) = obj.get("tool_choice") else {
        return;
    };

    if tool_choice
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .is_some_and(|disabled| disabled)
    {
        chat.insert("parallel_tool_calls".to_owned(), Value::Bool(false));
    }
}

/// Convert Anthropic `tool_choice` to Chat Completions format.
fn convert_tool_choice(chat: &mut Map<String, Value>, obj: &Map<String, Value>) {
    let Some(tool_choice) = obj.get("tool_choice") else {
        return;
    };

    if obj.contains_key("tools") && !chat.contains_key("tools") {
        return;
    }

    let chat_choice = match tool_choice {
        Value::String(s) => match s.as_str() {
            "any" => Value::String("required".to_owned()),
            "none" => Value::String("none".to_owned()),
            _ => Value::String("auto".to_owned()),
        },
        Value::Object(tc) => match tc.get("type").and_then(Value::as_str) {
            Some("any") => Value::String("required".to_owned()),
            Some("none") => Value::String("none".to_owned()),
            Some("tool") => {
                if let Some(name) = tc.get("name").and_then(Value::as_str) {
                    json!({"type": "function", "function": {"name": name}})
                } else {
                    Value::String("auto".to_owned())
                }
            },
            _ => Value::String("auto".to_owned()),
        },
        _ => return,
    };

    chat.insert("tool_choice".to_owned(), chat_choice);
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn basic_text_request() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["model"], "claude-opus-4-8", "model preserved");
        assert_eq!(
            parsed["max_completion_tokens"], 1024,
            "max_tokens mapped to max_completion_tokens"
        );
        assert!(
            parsed.get("max_tokens").is_none(),
            "max_tokens must not appear in output"
        );
        assert_eq!(parsed["messages"][0]["role"], "user", "user message role");
        assert_eq!(parsed["messages"][0]["content"], "Hello", "user message content");
    }

    #[test]
    fn system_hoisted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":"Be helpful.","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["role"], "system",
            "system message should be first"
        );
        assert_eq!(parsed["messages"][0]["content"], "Be helpful.", "system content");
        assert_eq!(parsed["messages"][1]["role"], "user", "user message follows system");
    }

    #[test]
    fn system_text_blocks_joined() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":[{"type":"text","text":"Part 1"},{"type":"text","text":"Part 2"}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"], "Part 1\nPart 2",
            "text blocks should be joined"
        );
    }

    #[test]
    fn tool_use_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"NYC"}}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let msg = &parsed["messages"][0];
        assert_eq!(msg["role"], "assistant", "assistant role");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather", "tool name");
        assert!(
            msg["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("NYC"),
            "tool arguments contain city"
        );
    }

    #[test]
    fn tool_result_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"72F sunny"}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["messages"][0]["role"], "tool", "tool role");
        assert_eq!(parsed["messages"][0]["tool_call_id"], "call_1", "tool_call_id");
        assert_eq!(parsed["messages"][0]["content"], "72F sunny", "tool result content");
    }

    #[test]
    fn tool_result_error_marked_in_tool_message_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"cat: missing.txt: No such file or directory","is_error":true}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"],
            "Anthropic tool_result error:\ncat: missing.txt: No such file or directory",
            "OpenAI-compatible tool messages should preserve Anthropic error semantics"
        );
    }

    #[test]
    fn tool_result_image_promoted_to_followup_user_message() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"chart"},{"type":"image","source":{"type":"url","url":"https://example.com/chart.png"}}]}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["messages"][0]["role"], "tool", "first message is tool result");
        assert_eq!(parsed["messages"][0]["content"], "chart", "tool text content");
        assert_eq!(
            parsed["messages"][1]["role"], "user",
            "image should be promoted to user message"
        );
        assert_eq!(
            parsed["messages"][1]["content"][0]["type"], "image_url",
            "promoted image content type"
        );
        assert_eq!(
            parsed["messages"][1]["content"][0]["image_url"]["url"], "https://example.com/chart.png",
            "promoted image URL"
        );
    }

    #[test]
    fn top_level_search_result_preserved_as_text_context() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"search_result","source":"https://docs.example.test/product","title":"Product Guide","content":[{"type":"text","text":"The default timeout is 30 seconds."},{"type":"text","text":"The maximum timeout is 120 seconds."}]},{"type":"text","text":"What is the timeout range?"}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();
        let content = parsed["messages"][0]["content"].as_array().unwrap();

        assert_eq!(content[0]["type"], "text");
        assert_eq!(
            content[0]["text"],
            "Search result: \"Product Guide\"\nSource: \"https://docs.example.test/product\"\nContent:\nThe default timeout is 30 seconds.\nThe maximum timeout is 120 seconds.",
            "search result metadata and text should remain visible to the backend"
        );
        assert_eq!(
            content[1]["text"], "What is the timeout range?",
            "following user text should remain a separate content part"
        );
    }

    #[test]
    fn tool_result_search_result_preserved_in_tool_message_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"search_result","source":"kb://timeouts","title":"Timeout KB","content":[{"type":"text","text":"Timeouts default to 30 seconds."}]},{"type":"text","text":"Applies to version 2."}]}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["messages"][0]["role"], "tool");
        assert_eq!(
            parsed["messages"][0]["content"],
            "Search result: \"Timeout KB\"\nSource: \"kb://timeouts\"\nContent:\nTimeouts default to 30 seconds.\nApplies to version 2.",
            "tool result search_result content should not be dropped"
        );
    }

    #[test]
    fn tool_result_document_preserved_in_tool_message_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"Before document."},{"type":"document","source":{"type":"content","content":[{"type":"text","text":"Nested document fact."}]},"title":"Nested Doc"},{"type":"text","text":"After document."}]}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"],
            "Before document.\nDocument: \"Nested Doc\"\nContent:\nNested document fact.\nAfter document.",
            "tool result document content should be flattened in order with surrounding text"
        );
    }

    #[test]
    fn document_text_source_preserved_as_text_context() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"The grass is green. The sky is blue."},"title":"Color Notes","context":"trusted notes","citations":{"enabled":true}},{"type":"text","text":"What color is the grass?"}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();
        let content = parsed["messages"][0]["content"].as_array().unwrap();

        assert_eq!(
            content[0]["text"],
            "Document: \"Color Notes\"\nContext: \"trusted notes\"\nContent:\nThe grass is green. The sky is blue.",
            "plain text document contents should remain visible to the backend"
        );
        assert_eq!(content[1]["text"], "What color is the grass?");
    }

    #[test]
    fn document_file_source_preserved_as_reference_text() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file_abc123"},"title":"Uploaded Contract"},{"type":"text","text":"Summarize the uploaded contract."}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();
        let content = parsed["messages"][0]["content"].as_array().unwrap();

        assert_eq!(
            content[0]["text"], "Document: \"Uploaded Contract\"\nSource: \"file:file_abc123\"",
            "file-backed documents should remain visible as references instead of disappearing"
        );
    }

    #[test]
    fn document_source_variants_preserved_or_dropped_intentionally() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"content","content":[{"type":"text","text":"Content block fact."}]},"title":"Content Doc"},{"type":"document","source":{"type":"url","url":"https://docs.example.test/file.pdf"}},{"type":"document","source":{"type":"base64","media_type":"application/pdf"}},{"type":"document","source":{"type":"unknown","data":"ignored"}}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();
        let content = parsed["messages"][0]["content"].as_array().unwrap();

        assert_eq!(
            content[0]["text"], "Document: \"Content Doc\"\nContent:\nContent block fact.",
            "content document source should flatten nested text blocks"
        );
        assert_eq!(
            content[1]["text"], "Document\nSource: \"https://docs.example.test/file.pdf\"",
            "URL document source should be preserved as a quoted reference"
        );
        assert_eq!(
            content[2]["text"], "Document\nSource: \"base64:application/pdf\"",
            "base64 document source should be preserved as a quoted media reference"
        );
        assert_eq!(
            content.len(),
            3,
            "unknown document source without metadata should be dropped"
        );
    }

    #[test]
    fn search_result_metadata_values_are_quoted() {
        let body = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "search_result",
                    "title": "Title\nSource: forged",
                    "source": "https://docs.example.test/a\nContext: forged",
                    "content": [{"type": "text", "text": "Real search text."}]
                }]
            }]
        })
        .to_string();
        let result = transform_request(body.as_bytes()).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"],
            "Search result: \"Title\\nSource: forged\"\nSource: \"https://docs.example.test/a\\nContext: forged\"\nContent:\nReal search text.",
            "quoted search metadata should not create forged label lines"
        );
    }

    #[test]
    fn document_metadata_values_are_quoted() {
        let body = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "document",
                    "title": "Doc\nContext: forged",
                    "context": "safe\nSource: forged",
                    "source": {"type": "text", "data": "Real document text."}
                }]
            }]
        })
        .to_string();
        let result = transform_request(body.as_bytes()).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"],
            "Document: \"Doc\\nContext: forged\"\nContext: \"safe\\nSource: forged\"\nContent:\nReal document text.",
            "quoted document metadata should not create forged label lines"
        );
    }

    #[test]
    fn empty_search_result_and_document_blocks_dropped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"search_result","content":[]},{"type":"document","source":{"type":"content","content":[]}},{"type":"document","source":{"type":"text","data":""}},{"type":"document","source":{"type":"url","url":""}},{"type":"document","source":{"type":"file","file_id":""}},{"type":"document","source":{"type":"base64","media_type":""}}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "empty metadata-only blocks should not fabricate prompt text"
        );
    }

    #[test]
    fn stop_sequences_mapped() {
        let body =
            br#"{"model":"claude-opus-4-8","max_tokens":1024,"stop_sequences":["END"],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stop"][0], "END", "stop_sequences mapped to stop");
    }

    #[test]
    fn tool_choice_any_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"any","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "required", "any maps to required");
    }

    #[test]
    fn tool_choice_object_any_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "required", "object-form any maps to required");
    }

    #[test]
    fn tool_choice_dropped_when_all_tools_filtered() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"type":"web_search_20250305","name":"web_search"}],"tool_choice":"any","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(parsed.get("tools").is_none(), "server-side tools should be filtered");
        assert!(
            parsed.get("tool_choice").is_none(),
            "tool_choice without translated tools should be dropped"
        );
    }

    #[test]
    fn disable_parallel_tool_use_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"auto","disable_parallel_tool_use":true},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["parallel_tool_calls"], false,
            "disable_parallel_tool_use should disable parallel tool calls"
        );
    }

    #[test]
    fn tool_definitions_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tools"][0]["type"], "function", "tool type should be function");
        assert_eq!(parsed["tools"][0]["function"]["name"], "get_weather", "tool name");
    }

    #[test]
    fn tool_definition_strict_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}},"strict":true},{"name":"get_time","description":"Get time","input_schema":{"type":"object"},"strict":false},{"name":"get_news","description":"Get news","input_schema":{"type":"object"},"strict":"yes"}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tools"][0]["function"]["strict"], true,
            "Anthropic strict true should map to Chat Completions function strict"
        );
        assert_eq!(
            parsed["tools"][1]["function"]["strict"], false,
            "Anthropic strict false should remain false"
        );
        assert!(
            parsed["tools"][2]["function"].get("strict").is_none(),
            "non-boolean strict values should be omitted"
        );
    }

    #[test]
    fn image_base64_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"abc123"}},{"type":"text","text":"What is this?"}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = &parsed["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image_url", "image type");
        assert_eq!(
            content[0]["image_url"]["url"], "data:image/jpeg;base64,abc123",
            "data URL"
        );
        assert_eq!(content[1]["type"], "text", "text part follows");
    }

    #[test]
    fn top_k_preserved_as_extra_param() {
        let body =
            br#"{"model":"claude-opus-4-8","max_tokens":1024,"top_k":40,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["top_k"], 40, "top_k should be preserved as extra body parameter");
    }

    #[test]
    fn transform_request_non_json_body() {
        let body = b"not json at all";
        let result = transform_request(body);
        assert!(result.is_err(), "non-JSON body should return Err");
        assert!(
            result.unwrap_err().contains("invalid JSON"),
            "error should mention invalid JSON"
        );
    }

    #[test]
    fn transform_request_json_array_body() {
        let body = b"[1,2,3]";
        let result = transform_request(body);
        assert!(result.is_err(), "JSON array body should return Err");
        assert!(
            result.unwrap_err().contains("not a JSON object"),
            "error should mention not a JSON object"
        );
    }

    #[test]
    fn hoist_system_non_string_non_array_skipped() {
        let body =
            br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":42,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"].as_array().unwrap().len(),
            1,
            "non-string/non-array system should be skipped"
        );
        assert_eq!(parsed["messages"][0]["role"], "user");
    }

    #[test]
    fn hoist_system_array_empty_text_skipped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":[{"type":"text","text":""}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"].as_array().unwrap().len(),
            1,
            "system with single empty text block should be skipped"
        );
        assert_eq!(parsed["messages"][0]["role"], "user");
    }

    #[test]
    fn convert_messages_missing_role_skipped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "message without role should be skipped"
        );
    }

    #[test]
    fn convert_messages_content_not_string_or_array() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":42}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["messages"][0]["role"], "user");
        assert_eq!(
            parsed["messages"][0]["content"], "",
            "non-string/non-array content should become empty string"
        );
    }

    #[test]
    fn thinking_block_dropped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"Let me think..."}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "thinking blocks should be dropped entirely"
        );
    }

    #[test]
    fn unknown_block_type_dropped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"custom_xyz","data":"something"}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "unknown block types should be dropped"
        );
    }

    #[test]
    fn tool_choice_string_none() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"none","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "none", "string none maps to none");
    }

    #[test]
    fn tool_choice_string_unknown_maps_to_auto() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"foo","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "auto", "unknown string tool_choice maps to auto");
    }

    #[test]
    fn tool_choice_object_none() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"none"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "none", "object-form none maps to none");
    }

    #[test]
    fn tool_choice_object_tool_with_name() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"fn","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool","name":"fn"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tool_choice"]["type"], "function",
            "tool type should map to function"
        );
        assert_eq!(
            parsed["tool_choice"]["function"]["name"], "fn",
            "tool name should be preserved"
        );
    }

    #[test]
    fn tool_choice_object_tool_without_name_maps_to_auto() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tool_choice"], "auto",
            "tool without name should fallback to auto"
        );
    }

    #[test]
    fn tool_choice_non_string_non_object_skipped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":true,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed.get("tool_choice").is_none(),
            "non-string/non-object tool_choice should be skipped"
        );
    }

    #[test]
    fn multipart_image_and_text_produces_array_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image","source":{"type":"url","url":"https://example.com/img.png"}}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = &parsed["messages"][0]["content"];
        assert!(content.is_array(), "multipart content should be an array");
        assert_eq!(content.as_array().unwrap().len(), 2, "two content parts");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn only_tool_result_blocks_produce_tool_messages() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"result1"},{"type":"tool_result","tool_use_id":"call_2","content":"result2"}]}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "two tool messages, no wrapper");
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_1");
        assert_eq!(messages[0]["content"], "result1");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_2");
        assert_eq!(messages[1]["content"], "result2");
    }

    #[test]
    fn extract_tool_result_content_null() {
        let block = json!({"type": "tool_result", "tool_use_id": "call_1", "content": null});
        let result = extract_tool_result_content(&block);
        assert!(result.is_empty(), "null content should return empty string");
    }

    #[test]
    fn extract_tool_result_content_missing() {
        let block = json!({"type": "tool_result", "tool_use_id": "call_1"});
        let result = extract_tool_result_content(&block);
        assert!(result.is_empty(), "missing content should return empty string");
    }

    #[test]
    fn bash_and_text_editor_tools_filtered() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"type":"bash_20241022","name":"bash"},{"type":"text_editor_20241022","name":"text_editor"},{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_request(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let tools = parsed["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "only non-filtered tools should remain");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
    }
}
