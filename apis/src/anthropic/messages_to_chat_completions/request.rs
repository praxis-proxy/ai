// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages to Chat Completions-compatible request transformation.

use serde_json::{Map, Value, json};
use tracing::warn;

use crate::json_body::{insert_if_some, take_string};

// -----------------------------------------------------------------------------
// Request Transformation
// -----------------------------------------------------------------------------

/// Transform a parsed Anthropic Messages request body into Chat
/// Completions-compatible format.
/// Returns the transformed JSON bytes, or an error message.
pub(crate) fn transform_request(value: Value) -> Result<Vec<u8>, String> {
    let Value::Object(mut body) = value else {
        return Err("request body is not a JSON object".to_owned());
    };

    // Take every mapped field up front, in one place. Each becomes an owned
    // local that is moved into the helper emitting it.
    let model = body.remove("model");
    let max_tokens = body.remove("max_tokens");
    let system = body.remove("system");
    let messages = body.remove("messages");
    let stream = body.remove("stream");
    let stream_options = body.remove("stream_options");
    let stop_sequences = body.remove("stop_sequences");
    let temperature = body.remove("temperature");
    let top_p = body.remove("top_p");
    let top_k = body.remove("top_k");
    let tools = body.remove("tools");
    let tool_choice = body.remove("tool_choice");
    let had_tools = tools.is_some();
    // Fields with no Chat Completions mapping are not forwarded. Dropping the
    // parsed body here makes any later read of it a compile error.
    drop(body);

    let mut chat = Map::new();
    insert_if_some(&mut chat, "model", model);
    chat.insert("messages".to_owned(), build_messages(system, messages));
    insert_if_some(&mut chat, "max_completion_tokens", max_tokens);
    convert_stream(&mut chat, stream, stream_options);
    map_parameters(&mut chat, stop_sequences, temperature, top_p, top_k);
    convert_tools(&mut chat, tools);
    convert_parallel_tool_calls(&mut chat, tool_choice.as_ref());
    convert_tool_choice(&mut chat, tool_choice, had_tools);

    serde_json::to_vec(&Value::Object(chat)).map_err(|e| format!("serialization failed: {e}"))
}

/// Build the Chat Completions `messages` array from the Anthropic `system` and
/// `messages` fields.
fn build_messages(system: Option<Value>, messages: Option<Value>) -> Value {
    let mut converted = Vec::new();
    hoist_system(&mut converted, system);
    convert_messages(&mut converted, messages);
    Value::Array(converted)
}

/// Build a Chat Completions message carrying plain string content.
fn text_message(role: String, content: String) -> Value {
    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String(role));
    message.insert("content".to_owned(), Value::String(content));
    Value::Object(message)
}

/// Build a Chat Completions `text` content part, moving `text` into it.
fn text_content_part(text: String) -> Value {
    let mut part = Map::new();
    part.insert("type".to_owned(), Value::String("text".to_owned()));
    part.insert("text".to_owned(), Value::String(text));
    Value::Object(part)
}

/// Build a Chat Completions `image_url` content part, moving `url` into it.
fn image_content_part(url: String) -> Value {
    let mut image_url = Map::new();
    image_url.insert("url".to_owned(), Value::String(url));

    let mut part = Map::new();
    part.insert("type".to_owned(), Value::String("image_url".to_owned()));
    part.insert("image_url".to_owned(), Value::Object(image_url));
    Value::Object(part)
}

// -----------------------------------------------------------------------------
// System Message Hoisting
// -----------------------------------------------------------------------------

/// Hoist Anthropic top-level `system` to a Chat Completions system message.
fn hoist_system(messages: &mut Vec<Value>, system: Option<Value>) {
    let content = match system {
        Some(Value::String(text)) => text,
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let Value::Object(mut block) = block
                    && let Some(text) = take_string(&mut block, "text")
                {
                    parts.push(text);
                }
            }
            parts.join("\n")
        },
        _ => return,
    };

    if !content.is_empty() {
        messages.push(text_message("system".to_owned(), content));
    }
}

// -----------------------------------------------------------------------------
// Message Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic messages array to Chat Completions messages.
fn convert_messages(messages: &mut Vec<Value>, source: Option<Value>) {
    let Some(Value::Array(anthropic_messages)) = source else {
        return;
    };

    for msg in anthropic_messages {
        let Value::Object(mut msg) = msg else {
            continue;
        };
        let Some(role) = take_string(&mut msg, "role") else {
            continue;
        };

        match msg.remove("content") {
            Some(Value::String(text)) => {
                messages.push(text_message(role, text));
            },
            Some(Value::Array(blocks)) => {
                convert_content_blocks(messages, &role, blocks);
            },
            _ => {
                messages.push(text_message(role, String::new()));
            },
        }
    }
}

/// Convert typed content blocks to Chat Completions-compatible format.
/// Consumes the blocks to move their payloads into the translated message.
fn convert_content_blocks(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        let mut block = match block {
            Value::Object(block) => block,
            _ => Map::new(),
        };
        let block_type = take_string(&mut block, "type").unwrap_or_default();
        convert_single_block(block, &block_type, messages, role, &mut content_parts, &mut tool_calls);
    }

    finalize_content_blocks(messages, role, &mut content_parts, tool_calls);
}

/// Process a single content block within a message.
#[expect(
    clippy::too_many_arguments,
    reason = "accumulator pattern requires passing all state"
)]
fn convert_single_block(
    block: Map<String, Value>,
    block_type: &str,
    messages: &mut Vec<Value>,
    role: &str,
    content_parts: &mut Vec<Value>,
    tool_calls: &mut Vec<Value>,
) {
    match block_type {
        "text" => convert_text_block(block, content_parts),
        "image" => convert_image_block(block, content_parts),
        "search_result" => convert_search_result_block(block, content_parts),
        "document" => convert_document_block(block, content_parts),
        "tool_use" => convert_tool_use_block(block, tool_calls),
        "tool_result" => {
            flush_content_parts(messages, content_parts, role);
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
fn convert_text_block(mut block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(text) = take_string(&mut block, "text") {
        content_parts.push(text_content_part(text));
    }
}

/// Convert an image content block.
fn convert_image_block(mut block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(source) = block.remove("source")
        && let Some(url_val) = convert_image_source(source)
    {
        content_parts.push(image_content_part(url_val));
    }
}

/// Convert a `search_result` block to backend-visible text context.
fn convert_search_result_block(block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(text) = flatten_search_result(block) {
        content_parts.push(text_content_part(text));
    }
}

/// Convert a `document` block to backend-visible text context.
fn convert_document_block(block: Map<String, Value>, content_parts: &mut Vec<Value>) {
    if let Some(text) = flatten_document(block) {
        content_parts.push(text_content_part(text));
    }
}

/// Take the string out of a `text` content part, leaving other parts untouched.
fn take_text_part(part: &mut Value) -> Option<String> {
    let part = part.as_object_mut()?;
    if part.get("type").and_then(Value::as_str) != Some("text") {
        return None;
    }
    take_string(part, "text")
}

/// Concatenate the text of every text content part, moving each string out.
fn take_joined_text(content_parts: &mut [Value]) -> Option<String> {
    let mut joined: Option<String> = None;
    for part in content_parts {
        let Some(text) = take_text_part(part) else {
            continue;
        };
        match &mut joined {
            None => joined = Some(text),
            Some(acc) => acc.push_str(&text),
        }
    }
    joined
}

/// Convert a `tool_use` content block to a Chat Completions tool call.
fn convert_tool_use_block(mut block: Map<String, Value>, tool_calls: &mut Vec<Value>) {
    let id = take_string(&mut block, "id").unwrap_or_default();
    let name = take_string(&mut block, "name").unwrap_or_default();

    let args = match block.remove("input") {
        Some(input) => serde_json::to_string(&input).unwrap_or_default(),
        None => serde_json::to_string(&Value::Object(Map::new())).unwrap_or_default(),
    };

    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name));
    function.insert("arguments".to_owned(), Value::String(args));

    let mut tool_call = Map::new();
    tool_call.insert("id".to_owned(), Value::String(id));
    tool_call.insert("type".to_owned(), Value::String("function".to_owned()));
    tool_call.insert("function".to_owned(), Value::Object(function));
    tool_calls.push(Value::Object(tool_call));
}

/// Convert a `tool_result` content block to a Chat Completions tool message.
fn convert_tool_result_block(mut block: Map<String, Value>, messages: &mut Vec<Value>) {
    let tool_call_id = take_string(&mut block, "tool_use_id").unwrap_or_default();
    let is_error = block.get("is_error").and_then(Value::as_bool) == Some(true);

    let (mut result_content, image_content) = split_tool_result_content(block.remove("content"));

    if is_error {
        result_content = mark_tool_result_error(result_content);
    }

    let mut tool_message = Map::new();
    tool_message.insert("role".to_owned(), Value::String("tool".to_owned()));
    tool_message.insert("tool_call_id".to_owned(), Value::String(tool_call_id));
    tool_message.insert("content".to_owned(), Value::String(result_content));
    messages.push(Value::Object(tool_message));

    if !image_content.is_empty() {
        let mut image_message = Map::new();
        image_message.insert("role".to_owned(), Value::String("user".to_owned()));
        image_message.insert("content".to_owned(), Value::Array(image_content));
        messages.push(Value::Object(image_message));
    }
}

/// Emit the final message for accumulated content and tool calls.
fn finalize_content_blocks(
    messages: &mut Vec<Value>,
    role: &str,
    content_parts: &mut Vec<Value>,
    tool_calls: Vec<Value>,
) {
    if role == "assistant" && !tool_calls.is_empty() {
        let mut msg = Map::new();
        msg.insert("role".to_owned(), Value::String("assistant".to_owned()));
        if let Some(text) = take_joined_text(content_parts) {
            msg.insert("content".to_owned(), Value::String(text));
        }
        msg.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        messages.push(Value::Object(msg));
    } else {
        flush_content_parts(messages, content_parts, role);
    }
}

/// Flush accumulated content parts as a message.
fn flush_content_parts(messages: &mut Vec<Value>, content_parts: &mut Vec<Value>, role: &str) {
    if content_parts.is_empty() {
        return;
    }

    let lone_text = match content_parts.as_mut_slice() {
        [part] => take_text_part(part),
        _ => None,
    };

    if let Some(text) = lone_text {
        messages.push(text_message(role.to_owned(), text));
    } else {
        let mut msg = Map::new();
        msg.insert("role".to_owned(), Value::String(role.to_owned()));
        msg.insert("content".to_owned(), Value::Array(std::mem::take(content_parts)));
        messages.push(Value::Object(msg));
    }

    content_parts.clear();
}

// -----------------------------------------------------------------------------
// Image Source Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic image source to an `image_url` URL string consuming the
/// source.
fn convert_image_source(source: Value) -> Option<String> {
    let Value::Object(mut source) = source else {
        return None;
    };
    let source_type = take_string(&mut source, "type")?;

    match source_type.as_str() {
        "base64" => {
            let media_type = take_string(&mut source, "media_type")?;
            let data = take_string(&mut source, "data")?;
            Some(format!("data:{media_type};base64,{data}"))
        },
        "url" => take_string(&mut source, "url"),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Tool Result Content Extraction
// -----------------------------------------------------------------------------

/// Split a `tool_result` block's `content` into its flattened text and the
/// image parts promoted to a follow-up user message.
fn split_tool_result_content(content: Option<Value>) -> (String, Vec<Value>) {
    match content {
        Some(Value::String(text)) => (text, Vec::new()),
        Some(Value::Array(parts)) => split_tool_result_parts(parts),
        _ => (String::new(), Vec::new()),
    }
}

/// Split the parts of an array-form `tool_result.content`.
fn split_tool_result_parts(parts: Vec<Value>) -> (String, Vec<Value>) {
    let mut text_parts = Vec::new();
    let mut image_parts = Vec::new();

    for part in parts {
        let Value::Object(mut part) = part else {
            continue;
        };
        // No branch below reads `type` again, so it is taken out with the rest.
        match take_string(&mut part, "type").as_deref() {
            Some("text") => {
                if let Some(text) = take_string(&mut part, "text") {
                    text_parts.push(text);
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
            Some("image") => convert_image_block(part, &mut image_parts),
            _ => {},
        }
    }

    (text_parts.join("\n"), image_parts)
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

/// Flatten an Anthropic `search_result` block to plain text.
fn flatten_search_result(mut block: Map<String, Value>) -> Option<String> {
    let title = take_string(&mut block, "title").filter(|title| !title.is_empty());
    let source = take_string(&mut block, "source").filter(|source| !source.is_empty());
    let content = extract_text_blocks(block.remove("content"));

    if title.is_none() && source.is_none() && content.is_empty() {
        return None;
    }

    let mut flattened = String::new();

    if let Some(title) = title {
        flattened.push_str("Search result: ");
        flattened.push_str(&quote_label_value(&title));
    } else {
        flattened.push_str("Search result");
    }

    if let Some(source) = source {
        flattened.push_str("\nSource: ");
        flattened.push_str(&quote_label_value(&source));
    }

    if !content.is_empty() {
        flattened.push_str("\nContent:");
        for text in content {
            flattened.push('\n');
            flattened.push_str(&text);
        }
    }

    Some(flattened)
}

/// Flatten an Anthropic `document` block to plain text.
fn flatten_document(mut block: Map<String, Value>) -> Option<String> {
    let title = take_string(&mut block, "title").filter(|title| !title.is_empty());
    let context = take_string(&mut block, "context").filter(|context| !context.is_empty());
    let source_text = flatten_document_source(block.remove("source"));

    if title.is_none() && context.is_none() && source_text.is_none() {
        return None;
    }

    let mut flattened = String::new();

    if let Some(title) = title {
        flattened.push_str("Document: ");
        flattened.push_str(&quote_label_value(&title));
    } else {
        flattened.push_str("Document");
    }

    if let Some(context) = context {
        flattened.push_str("\nContext: ");
        flattened.push_str(&quote_label_value(&context));
    }

    if let Some(source_text) = source_text {
        flattened.push('\n');
        flattened.push_str(&source_text);
    }

    Some(flattened)
}

/// Flatten a `document.source` value to extractable text or a stable reference.
fn flatten_document_source(source: Option<Value>) -> Option<String> {
    let Value::Object(mut source) = source? else {
        return None;
    };
    let source_type = take_string(&mut source, "type")?;

    match source_type.as_str() {
        "text" => take_string(&mut source, "data")
            .filter(|data| !data.is_empty())
            .map(|data| format!("Content:\n{data}")),
        "content" => {
            let lines = extract_text_blocks(source.remove("content"));
            non_empty_lines(&lines).map(|content| format!("Content:\n{content}"))
        },
        "url" => take_string(&mut source, "url")
            .filter(|url| !url.is_empty())
            .map(|url| format!("Source: {}", quote_label_value(&url))),
        "file" => take_string(&mut source, "file_id")
            .filter(|file_id| !file_id.is_empty())
            .map(|file_id| format!("Source: {}", quote_label_value(&format!("file:{file_id}")))),
        "base64" => take_string(&mut source, "media_type")
            .filter(|media_type| !media_type.is_empty())
            .map(|media_type| format!("Source: {}", quote_label_value(&format!("base64:{media_type}")))),
        _ => None,
    }
}

/// Quote metadata values so embedded newlines cannot forge flattening labels.
fn quote_label_value(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

/// Extract text from an array of Anthropic text blocks, moving each string out.
fn extract_text_blocks(value: Option<Value>) -> Vec<String> {
    let Some(Value::Array(blocks)) = value else {
        return Vec::new();
    };

    let mut texts = Vec::new();
    for block in blocks {
        let Value::Object(mut block) = block else {
            continue;
        };
        if block.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        if let Some(text) = take_string(&mut block, "text")
            && !text.is_empty()
        {
            texts.push(text);
        }
    }
    texts
}

/// Join lines if at least one line contains content.
fn non_empty_lines(lines: &[String]) -> Option<String> {
    lines.iter().any(|line| !line.is_empty()).then(|| lines.join("\n"))
}

// -----------------------------------------------------------------------------
// Parameter Mapping
// -----------------------------------------------------------------------------

/// Move `stream` through and request streaming usage when enabled.
fn convert_stream(chat: &mut Map<String, Value>, stream: Option<Value>, stream_options: Option<Value>) {
    let Some(stream) = stream else {
        return;
    };

    let streaming = stream.as_bool() == Some(true);
    chat.insert("stream".to_owned(), stream);

    if !streaming {
        return;
    }

    let mut opts = match stream_options {
        Some(Value::Object(opts)) => opts,
        _ => Map::new(),
    };
    opts.insert("include_usage".to_owned(), Value::Bool(true));
    chat.insert("stream_options".to_owned(), Value::Object(opts));
}

/// Map Anthropic parameters to Chat Completions-compatible equivalents.
///
/// `top_k` has no standard Chat Completions equivalent but is preserved
/// as an extra body parameter for backends that support it
/// (e.g. vLLM).
fn map_parameters(
    chat: &mut Map<String, Value>,
    stop_sequences: Option<Value>,
    temperature: Option<Value>,
    top_p: Option<Value>,
    top_k: Option<Value>,
) {
    insert_if_some(chat, "stop", stop_sequences);
    insert_if_some(chat, "temperature", temperature);
    insert_if_some(chat, "top_p", top_p);
    insert_if_some(chat, "top_k", top_k);
}

// -----------------------------------------------------------------------------
// Tool Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic tool definitions to Chat Completions function tools.
fn convert_tools(chat: &mut Map<String, Value>, tools: Option<Value>) {
    let Some(Value::Array(tools)) = tools else {
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

/// Return a stable JSON type name for diagnostics, never the value itself.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Classify an Anthropic tool: keep only untyped or explicit `custom` client
/// tools, dropping (and logging) every typed server tool so unknown server
/// tools fail closed instead of leaking to the backend as client functions.
fn is_translatable_client_tool(tool: &Value) -> bool {
    match tool.get("type") {
        None => true,
        Some(Value::String(tool_type)) if tool_type == "custom" => true,
        Some(Value::String(tool_type)) => {
            warn!(tool_type, "dropping typed Anthropic tool");
            false
        },
        Some(other) => {
            // Log only the JSON value kind, never the value itself: `type` is
            // attacker-controlled and could carry a large or sensitive payload.
            warn!(
                type_kind = json_type_name(other),
                "dropping Anthropic tool with non-string type"
            );
            false
        },
    }
}

/// Convert one Anthropic client tool definition to a Chat Completions tool.
///
/// Consumes the definition so `input_schema` — the largest value in a typical
/// agentic request — moves into the generated function parameters.
fn convert_tool_definition(tool: Value) -> Option<Value> {
    if !is_translatable_client_tool(&tool) {
        return None;
    }

    // A non-object tool entry carries no fields, so it translates to an empty function.
    let mut tool = match tool {
        Value::Object(fields) => fields,
        _ => Map::new(),
    };

    let name = take_string(&mut tool, "name").unwrap_or_default();
    let description = take_string(&mut tool, "description").unwrap_or_default();
    let parameters = tool.remove("input_schema").unwrap_or_else(|| json!({"type": "object"}));
    let strict = tool.get("strict").and_then(Value::as_bool);

    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name));
    function.insert("description".to_owned(), Value::String(description));
    function.insert("parameters".to_owned(), parameters);
    if let Some(strict) = strict {
        function.insert("strict".to_owned(), Value::Bool(strict));
    }

    // Built by hand rather than with `json!`, which would deep-clone the
    // function map back through the serializer.
    let mut chat_tool = Map::new();
    chat_tool.insert("type".to_owned(), Value::String("function".to_owned()));
    chat_tool.insert("function".to_owned(), Value::Object(function));
    Some(Value::Object(chat_tool))
}

// -----------------------------------------------------------------------------
// Tool Choice Conversion
// -----------------------------------------------------------------------------

/// Convert Anthropic `disable_parallel_tool_use` to Chat Completions format.
fn convert_parallel_tool_calls(chat: &mut Map<String, Value>, tool_choice: Option<&Value>) {
    let Some(Value::Object(tool_choice)) = tool_choice else {
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
fn convert_tool_choice(chat: &mut Map<String, Value>, tool_choice: Option<Value>, had_tools: bool) {
    let Some(tool_choice) = tool_choice else {
        return;
    };

    if had_tools && !chat.contains_key("tools") {
        return;
    }

    let chat_choice = match tool_choice {
        Value::String(keyword) => Value::String(tool_choice_keyword(&keyword).to_owned()),
        Value::Object(tool_choice) => object_tool_choice(tool_choice),
        _ => return,
    };

    chat.insert("tool_choice".to_owned(), chat_choice);
}

/// Map an Anthropic `tool_choice` type to its Chat Completions keyword.
fn tool_choice_keyword(anthropic: &str) -> &'static str {
    match anthropic {
        "any" => "required",
        "none" => "none",
        _ => "auto",
    }
}

/// Convert an object-form `tool_choice`, moving a named tool's name through.
fn object_tool_choice(mut tool_choice: Map<String, Value>) -> Value {
    let names_a_tool = tool_choice.get("type").and_then(Value::as_str) == Some("tool");

    if names_a_tool && let Some(name) = take_string(&mut tool_choice, "name") {
        let mut function = Map::new();
        function.insert("name".to_owned(), Value::String(name));

        let mut choice = Map::new();
        choice.insert("type".to_owned(), Value::String("function".to_owned()));
        choice.insert("function".to_owned(), Value::Object(function));
        return Value::Object(choice);
    }

    // A `tool` choice without a usable name degrades to `auto`.
    let kind = tool_choice.get("type").and_then(Value::as_str).unwrap_or_default();
    Value::String(tool_choice_keyword(kind).to_owned())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    /// Parse a raw request body and transform it, mirroring the filter's
    /// parse-once call path including its parse-error message.
    fn transform_bytes(body: &[u8]) -> Result<Vec<u8>, String> {
        let value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
        transform_request(value)
    }

    #[test]
    fn basic_text_request() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}"#;
        let result = transform_bytes(body).unwrap();
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
    fn mapped_fields_keep_a_stable_serialized_key_order() {
        // `serde_json` runs with `preserve_order`, so the order fields are
        // emitted in `transform_request` is the order sent upstream. Pin it so
        // reordering the emission sequence cannot silently reshape the wire body.
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":true,"stop_sequences":["x"],"temperature":0.5,"top_p":0.9,"top_k":40,"tools":[{"name":"t","input_schema":{"type":"object"}}],"tool_choice":{"type":"any","disable_parallel_tool_use":true},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let keys: Vec<&str> = parsed.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec![
                "model",
                "messages",
                "max_completion_tokens",
                "stream",
                "stream_options",
                "stop",
                "temperature",
                "top_p",
                "top_k",
                "tools",
                "parallel_tool_calls",
                "tool_choice",
            ],
            "translated request key order must stay stable"
        );
    }

    #[test]
    fn untranslated_top_level_fields_are_dropped() {
        let body = br#"{"model":"m","metadata":{"user_id":"u"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed.get("metadata").is_none(),
            "fields with no Chat Completions mapping are not forwarded"
        );
    }

    #[test]
    fn tool_input_schema_is_preserved_verbatim() {
        let body = br#"{"model":"m","tools":[{"name":"t","description":"d","input_schema":{"type":"object","properties":{"a":{"type":"array","items":{"type":"string"}},"b":{"enum":[1,2,3]}},"required":["a"],"additionalProperties":false}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tools"][0]["function"]["parameters"],
            json!({
                "type": "object",
                "properties": {"a": {"type": "array", "items": {"type": "string"}}, "b": {"enum": [1, 2, 3]}},
                "required": ["a"],
                "additionalProperties": false
            }),
            "moving the schema must not alter its contents"
        );
    }

    #[test]
    fn tool_definition_with_non_string_name_falls_back_to_empty() {
        let body = br#"{"model":"m","tools":[{"name":42,"description":true,"input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tools"][0]["function"]["name"], "",
            "non-string name yields empty"
        );
        assert_eq!(
            parsed["tools"][0]["function"]["description"], "",
            "non-string description yields empty"
        );
    }

    #[test]
    fn stream_options_dropped_when_streaming_disabled() {
        let body = br#"{"model":"m","stream":false,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stream"], false);
        assert!(
            parsed.get("stream_options").is_none(),
            "stream_options is meaningless without streaming"
        );
    }

    #[test]
    fn caller_stream_options_are_preserved_alongside_include_usage() {
        let body =
            br#"{"model":"m","stream":true,"stream_options":{"custom":1},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["stream_options"]["custom"], 1,
            "caller options are moved through"
        );
        assert_eq!(parsed["stream_options"]["include_usage"], true);
    }

    #[test]
    fn system_hoisted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":"Be helpful.","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"], "Part 1\nPart 2",
            "text blocks should be joined"
        );
    }

    #[test]
    fn tool_use_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"NYC"}}]}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["messages"][0]["role"], "tool", "tool role");
        assert_eq!(parsed["messages"][0]["tool_call_id"], "call_1", "tool_call_id");
        assert_eq!(parsed["messages"][0]["content"], "72F sunny", "tool result content");
    }

    #[test]
    fn tool_result_error_marked_in_tool_message_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"cat: missing.txt: No such file or directory","is_error":true}]}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body.as_bytes()).unwrap();
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
        let result = transform_bytes(body.as_bytes()).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stop"][0], "END", "stop_sequences mapped to stop");
    }

    #[test]
    fn tool_choice_any_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"any","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "required", "any maps to required");
    }

    #[test]
    fn tool_choice_object_any_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "required", "object-form any maps to required");
    }

    #[test]
    fn tool_choice_dropped_when_all_tools_filtered() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"type":"web_search_20250305","name":"web_search"}],"tool_choice":"any","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["parallel_tool_calls"], false,
            "disable_parallel_tool_use should disable parallel tool calls"
        );
    }

    #[test]
    fn tool_definitions_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tools"][0]["type"], "function", "tool type should be function");
        assert_eq!(parsed["tools"][0]["function"]["name"], "get_weather", "tool name");
    }

    #[test]
    fn tool_definition_strict_mapped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}},"strict":true},{"name":"get_time","description":"Get time","input_schema":{"type":"object"},"strict":false},{"name":"get_news","description":"Get news","input_schema":{"type":"object"},"strict":"yes"}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["top_k"], 40, "top_k should be preserved as extra body parameter");
    }

    #[test]
    fn transform_request_non_json_body() {
        let body = b"not json at all";
        let result = transform_bytes(body);
        assert!(result.is_err(), "non-JSON body should return Err");
        assert!(
            result.unwrap_err().contains("invalid JSON"),
            "error should mention invalid JSON"
        );
    }

    #[test]
    fn transform_request_json_array_body() {
        let body = b"[1,2,3]";
        let result = transform_bytes(body);
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "message without role should be skipped"
        );
    }

    #[test]
    fn convert_messages_content_not_string_or_array() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":42}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "thinking blocks should be dropped entirely"
        );
    }

    #[test]
    fn unknown_block_type_dropped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"custom_xyz","data":"something"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed["messages"].as_array().unwrap().is_empty(),
            "unknown block types should be dropped"
        );
    }

    #[test]
    fn tool_choice_string_none() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"none","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "none", "string none maps to none");
    }

    #[test]
    fn tool_choice_string_unknown_maps_to_auto() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":"foo","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "auto", "unknown string tool_choice maps to auto");
    }

    #[test]
    fn tool_choice_object_none() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"none"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["tool_choice"], "none", "object-form none maps to none");
    }

    #[test]
    fn tool_choice_object_tool_with_name() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"name":"fn","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool","name":"fn"},"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["tool_choice"], "auto",
            "tool without name should fallback to auto"
        );
    }

    #[test]
    fn tool_choice_non_string_non_object_skipped() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tool_choice":true,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed.get("tool_choice").is_none(),
            "non-string/non-object tool_choice should be skipped"
        );
    }

    #[test]
    fn multipart_image_and_text_produces_array_content() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this"},{"type":"image","source":{"type":"url","url":"https://example.com/img.png"}}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = &parsed["messages"][0]["content"];
        assert!(content.is_array(), "multipart content should be an array");
        assert_eq!(content.as_array().unwrap().len(), 2, "two content parts");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn two_text_blocks_stay_two_content_parts() {
        // String content is emitted only for a *single* text part. Two text
        // blocks keep their part boundaries rather than being joined.
        let body = br#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"one"},{"type":"text","text":"two"}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let content = parsed["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2, "two text blocks stay two parts");
        assert_eq!(content[0]["text"], "one");
        assert_eq!(content[1]["text"], "two");
    }

    #[test]
    fn assistant_tool_calls_join_all_text_blocks() {
        // The assistant+tool_calls branch joins every text part into one string,
        // a different rule from the single-part case above.
        let body = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"text","text":"one"},{"type":"text","text":"two"},{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let msg = &parsed["messages"][0];
        assert_eq!(msg["content"], "onetwo", "assistant text parts are joined");
        assert_eq!(msg["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn assistant_tool_calls_distinguish_empty_text_from_no_text() {
        // An empty text block still emits `content: ""`; no text block at all
        // emits no `content` key.
        let empty = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"text","text":""},{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let parsed: Value = serde_json::from_slice(&transform_bytes(empty).unwrap()).unwrap();
        assert_eq!(
            parsed["messages"][0]["content"], "",
            "an empty text block still emits content"
        );

        let none = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let parsed: Value = serde_json::from_slice(&transform_bytes(none).unwrap()).unwrap();
        assert!(
            parsed["messages"][0].get("content").is_none(),
            "no text block emits no content key"
        );
    }

    #[test]
    fn assistant_tool_calls_drop_image_parts() {
        // The joined-string branch cannot carry an image part, so it is dropped
        // while the surrounding text is still joined.
        let body = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"text","text":"one"},{"type":"image","source":{"type":"url","url":"https://example.com/i.png"}},{"type":"text","text":"two"},{"type":"tool_use","id":"c1","name":"f","input":{}}]}]}"#;
        let parsed: Value = serde_json::from_slice(&transform_bytes(body).unwrap()).unwrap();

        assert_eq!(
            parsed["messages"][0]["content"], "onetwo",
            "text joins across the dropped image"
        );
    }

    #[test]
    fn tool_use_without_input_serializes_an_empty_object() {
        let body = br#"{"model":"m","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"f"},{"type":"tool_use","id":"c2","name":"g","input":null}]}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let calls = parsed["messages"][0]["tool_calls"].as_array().unwrap();
        assert_eq!(
            calls[0]["function"]["arguments"], "{}",
            "absent input becomes an empty object"
        );
        assert_eq!(
            calls[1]["function"]["arguments"], "null",
            "an explicit null input is preserved as null"
        );
    }

    #[test]
    fn only_tool_result_blocks_produce_tool_messages() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"result1"},{"type":"tool_result","tool_use_id":"call_2","content":"result2"}]}]}"#;
        let result = transform_bytes(body).unwrap();
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
        let (text, images) = split_tool_result_content(Some(Value::Null));
        assert!(text.is_empty(), "null content should return empty string");
        assert!(images.is_empty(), "null content carries no images");
    }

    #[test]
    fn extract_tool_result_content_missing() {
        let (text, images) = split_tool_result_content(None);
        assert!(text.is_empty(), "missing content should return empty string");
        assert!(images.is_empty(), "missing content carries no images");
    }

    #[test]
    fn tool_result_content_split_keeps_text_and_images_in_one_pass() {
        let content = json!([
            {"type": "text", "text": "before"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "after"},
            {"type": "thinking", "thinking": "ignored"},
            "not an object"
        ]);

        let (text, images) = split_tool_result_content(Some(content));

        assert_eq!(text, "before\nafter", "text parts join in order, skipping non-text");
        assert_eq!(images.len(), 1, "one image part promoted");
        assert_eq!(images[0]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn only_client_tools_converted() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"tools":[{"type":"bash_20241022","name":"bash"},{"type":"text_editor_20241022","name":"text_editor"},{"type":"code_execution_20250522","name":"code_execution"},{"type":"computer_20250124","name":"computer","display_width_px":1024,"display_height_px":768},{"type":"future_server_tool_20270101","name":"future_server_tool"},{"type":42,"name":"invalid_type","input_schema":{"type":"object"}},{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{}}},{"type":"custom","name":"get_time","description":"Get time","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        let tools = parsed["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2, "only untyped and custom client tools should remain");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[1]["function"]["name"], "get_time");
    }

    #[test]
    fn streaming_request_includes_usage_option() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stream"], true, "stream should be true");
        assert_eq!(
            parsed["stream_options"]["include_usage"], true,
            "stream_options.include_usage should be set"
        );
    }

    #[test]
    fn non_streaming_request_omits_stream_options() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(
            parsed.get("stream_options").is_none(),
            "stream_options should not be present without stream:true"
        );
    }

    #[test]
    fn stream_false_omits_stream_options() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = transform_bytes(body).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stream"], false, "stream should be false");
        assert!(
            parsed.get("stream_options").is_none(),
            "stream_options should not be present when stream is false"
        );
    }
}
