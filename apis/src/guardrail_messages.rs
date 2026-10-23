// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-aware extraction of messages for `NeMo` guardrail checks.
//!
//! This module deliberately does not classify request bodies. Callers must
//! provide the request-head identity published by `ai_operation`.

use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::{
    anthropic::routes::is_create_message,
    openai::{
        chat_completions::routes::is_create_chat_completion,
        responses::routes::is_create_response,
        translation::{chat_completions::responses_request_to_chat_request, reasoning::ReasoningOptions},
    },
    operation::Transport,
    operation_classifier::AiOperationMatch,
};

/// Whether the body belongs to the client request or upstream response.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardrailMessageDirection {
    /// Extract input messages before an inference call.
    Request,
    /// Extract output messages after a non-streaming inference call.
    Response,
}

/// Failure while converting a protocol body to `NeMo` messages.
#[derive(Debug, Error)]
pub enum GuardrailMessageExtractError {
    /// The operation is not one of the inference operations supported here.
    #[error("unsupported guardrail operation {protocol}/{operation} for {direction:?}")]
    UnsupportedOperation {
        /// Protocol identity from the operation registry.
        protocol: &'static str,
        /// Operation identity from the operation registry.
        operation: &'static str,
        /// Extraction direction.
        direction: GuardrailMessageDirection,
    },
    /// The body was not valid JSON.
    #[error("guardrail body is not valid JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),
    /// The JSON did not satisfy the selected protocol contract.
    #[error("guardrail body has an invalid {0} shape")]
    InvalidShape(&'static str),
    /// A selected content item cannot be represented as text safely.
    #[error("guardrail body contains unsupported {0} content")]
    UnsupportedContent(&'static str),
}

/// Extract canonical `NeMo` messages using a trusted operation identity.
///
/// The operation match is produced from the request head by `ai_operation`.
/// This function never infers a protocol from JSON fields and never mutates
/// the supplied body.
///
/// # Errors
///
/// Returns [`GuardrailMessageExtractError`] when the operation is unsupported,
/// the body is invalid JSON, or it does not satisfy its selected protocol
/// contract.
#[expect(
    clippy::too_many_lines,
    reason = "dispatches each direction through registry-owned operation predicates"
)]
pub fn extract_guardrail_messages(
    operation: AiOperationMatch,
    direction: GuardrailMessageDirection,
    body: &[u8],
) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    if operation.transport != Transport::Http {
        return Err(GuardrailMessageExtractError::UnsupportedOperation {
            protocol: operation.application_protocol.as_str(),
            operation: operation.operation_id,
            direction,
        });
    }

    let json: Value = serde_json::from_slice(body).map_err(GuardrailMessageExtractError::InvalidJson)?;
    if is_create_chat_completion(operation.application_protocol, operation.operation_id) {
        return match direction {
            GuardrailMessageDirection::Request => extract_chat_request(json),
            GuardrailMessageDirection::Response => extract_chat_response(json),
        };
    }
    if is_create_response(operation.application_protocol, operation.operation_id) {
        return match direction {
            GuardrailMessageDirection::Request => extract_responses_request(&json),
            GuardrailMessageDirection::Response => extract_responses_response(json),
        };
    }
    if is_create_message(operation.application_protocol, operation.operation_id) {
        return match direction {
            GuardrailMessageDirection::Request => extract_anthropic_request(json),
            GuardrailMessageDirection::Response => extract_anthropic_response(json),
        };
    }

    Err(GuardrailMessageExtractError::UnsupportedOperation {
        protocol: operation.application_protocol.as_str(),
        operation: operation.operation_id,
        direction,
    })
}

/// Extract messages from a Chat Completions request.
fn extract_chat_request(mut json: Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let messages = json
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .ok_or(GuardrailMessageExtractError::InvalidShape(
            "Chat Completions request messages",
        ))?;
    validate_text_guardrail_messages(&messages, "Chat Completions request")?;
    Ok(messages)
}

/// Extract messages from a Chat Completions response.
fn extract_chat_response(mut json: Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let choices =
        json.get_mut("choices")
            .and_then(Value::as_array_mut)
            .ok_or(GuardrailMessageExtractError::InvalidShape(
                "Chat Completions response choices",
            ))?;
    if choices.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape(
            "Chat Completions response choices",
        ));
    }

    let messages = choices
        .iter_mut()
        .map(|choice| {
            choice
                .get_mut("message")
                .map(std::mem::take)
                .ok_or(GuardrailMessageExtractError::InvalidShape(
                    "Chat Completions response message",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_text_guardrail_messages(&messages, "Chat Completions response")?;
    Ok(messages)
}

/// Translate and extract text messages from an OpenAI Responses request.
fn extract_responses_request(json: &Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let translated = responses_request_to_chat_request(json, &ReasoningOptions::default())
        .map_err(|_error| GuardrailMessageExtractError::InvalidShape("OpenAI Responses request"))?;
    let messages = translated
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .cloned()
        .ok_or(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses request input",
        ))?;
    validate_text_guardrail_messages(&messages, "OpenAI Responses request")?;
    Ok(messages)
}

/// `NeMo`'s guardrail payload is text-oriented. Tool-call metadata is retained,
/// but media-bearing message content is rejected rather than forwarded to a
/// text-only evaluator.
fn validate_text_guardrail_messages(
    messages: &[Value],
    protocol: &'static str,
) -> Result<(), GuardrailMessageExtractError> {
    for message in messages {
        let Some(content) = message.get("content") else {
            continue;
        };
        match content {
            Value::Null | Value::String(_) => {},
            Value::Array(parts) => {
                for part in parts {
                    if part.get("type").and_then(Value::as_str) != Some("text") {
                        return Err(GuardrailMessageExtractError::UnsupportedContent(protocol));
                    }
                }
            },
            _ => {
                return Err(GuardrailMessageExtractError::UnsupportedContent(protocol));
            },
        }
    }
    Ok(())
}

/// Extract text and function calls from an OpenAI Responses response.
#[expect(
    clippy::too_many_lines,
    reason = "validates each supported OpenAI Responses output item in one pass"
)]
fn extract_responses_response(mut json: Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let output =
        json.get_mut("output")
            .and_then(Value::as_array_mut)
            .ok_or(GuardrailMessageExtractError::InvalidShape(
                "OpenAI Responses response output",
            ))?;
    if output.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses response output",
        ));
    }

    let mut messages = Vec::new();
    for item in output.iter() {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
        match item_type {
            "message" => {
                let role =
                    item.get("role")
                        .and_then(Value::as_str)
                        .ok_or(GuardrailMessageExtractError::InvalidShape(
                            "OpenAI Responses output message role",
                        ))?;
                if role != "assistant" {
                    return Err(GuardrailMessageExtractError::InvalidShape(
                        "OpenAI Responses output message role",
                    ));
                }
                let content = extract_responses_text(item.get("content"))?;
                if !content.is_empty() {
                    messages.push(json!({"role": role, "content": content}));
                }
            },
            "function_call" => {
                let call_id = required_string(item, "call_id", "OpenAI Responses function call")?;
                let name = required_string(item, "name", "OpenAI Responses function call")?;
                let arguments = required_string(item, "arguments", "OpenAI Responses function call")?;
                messages.push(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    }]
                }));
            },
            "reasoning" => {},
            _ => {
                return Err(GuardrailMessageExtractError::UnsupportedContent(
                    "OpenAI Responses output",
                ));
            },
        }
    }

    if messages.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses response text output",
        ));
    }
    Ok(messages)
}

/// Extract the textual content of an OpenAI Responses message.
fn extract_responses_text(content: Option<&Value>) -> Result<String, GuardrailMessageExtractError> {
    let parts = content
        .and_then(Value::as_array)
        .ok_or(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses message content",
        ))?;
    let mut text = String::new();
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => text.push_str(required_string(part, "text", "OpenAI Responses output text")?),
            Some("refusal") => text.push_str(required_string(part, "refusal", "OpenAI Responses refusal")?),
            Some(_) => {
                return Err(GuardrailMessageExtractError::UnsupportedContent(
                    "OpenAI Responses message",
                ));
            },
            None => {
                return Err(GuardrailMessageExtractError::InvalidShape(
                    "OpenAI Responses message content",
                ));
            },
        }
    }
    Ok(text)
}

/// Extract text messages from an Anthropic Messages request.
#[expect(
    clippy::too_many_lines,
    reason = "moves request content into canonical messages without copying strings"
)]
fn extract_anthropic_request(mut json: Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let object = json
        .as_object_mut()
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic request"))?;
    let mut messages = Vec::new();
    if let Some(system) = object.remove("system") {
        let content = extract_anthropic_text(system)?;
        if !content.is_empty() {
            messages.push(canonical_text_message("system".to_owned(), content));
        }
    }
    let input = object
        .remove("messages")
        .and_then(|messages| match messages {
            Value::Array(messages) => Some(messages),
            _ => None,
        })
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic request messages"))?;
    if input.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape("Anthropic request messages"));
    }
    for mut message in input {
        let message = message
            .as_object_mut()
            .ok_or(GuardrailMessageExtractError::InvalidShape(
                "Anthropic request message role",
            ))?;
        let role = take_required_string(message, "role", "Anthropic request message role")?;
        let content = message
            .remove("content")
            .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic text content"))?;
        messages.push(canonical_text_message(role, extract_anthropic_text(content)?));
    }
    Ok(messages)
}

/// Extract the assistant message from an Anthropic Messages response.
fn extract_anthropic_response(mut json: Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let object = json
        .as_object_mut()
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic response"))?;
    let role = take_required_string(object, "role", "Anthropic response role")?;
    if role != "assistant" {
        return Err(GuardrailMessageExtractError::InvalidShape("Anthropic response role"));
    }
    let content = object
        .remove("content")
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic text content"))?;
    let content = extract_anthropic_text(content)?;
    if content.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape("Anthropic response content"));
    }
    Ok(vec![canonical_text_message(role, content)])
}

/// Extract the textual content of an Anthropic Messages value.
fn extract_anthropic_text(value: Value) -> Result<String, GuardrailMessageExtractError> {
    let blocks = match value {
        Value::String(text) => return Ok(text),
        Value::Array(blocks) => blocks,
        _ => return Err(GuardrailMessageExtractError::InvalidShape("Anthropic text content")),
    };
    let mut text = String::new();
    for mut block in blocks {
        let block = block
            .as_object_mut()
            .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic content block"))?;
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(&take_required_string(block, "text", "Anthropic text block")?),
            Some(_) => return Err(GuardrailMessageExtractError::UnsupportedContent("Anthropic message")),
            None => return Err(GuardrailMessageExtractError::InvalidShape("Anthropic content block")),
        }
    }
    Ok(text)
}

/// Build a canonical text message while retaining owned protocol strings.
fn canonical_text_message(role: String, content: String) -> Value {
    Value::Object(Map::from_iter([
        ("role".to_owned(), Value::String(role)),
        ("content".to_owned(), Value::String(content)),
    ]))
}

/// Remove a required string field from an object.
fn take_required_string(
    object: &mut Map<String, Value>,
    field: &'static str,
    shape: &'static str,
) -> Result<String, GuardrailMessageExtractError> {
    match object.remove(field) {
        Some(Value::String(value)) => Ok(value),
        _ => Err(GuardrailMessageExtractError::InvalidShape(shape)),
    }
}

/// Return a required string field or report the selected protocol shape.
fn required_string<'a>(
    object: &'a Value,
    field: &'static str,
    shape: &'static str,
) -> Result<&'a str, GuardrailMessageExtractError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or(GuardrailMessageExtractError::InvalidShape(shape))
}

#[cfg(test)]
#[expect(clippy::indexing_slicing, reason = "tests assert known extractor output")]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;
    use crate::operation::{ApplicationProtocol, PathParameterOffsets, RequestBody};

    fn operation(protocol: &'static str, operation_id: &'static str) -> AiOperationMatch {
        AiOperationMatch {
            application_protocol: ApplicationProtocol::new(protocol),
            operation_id,
            transport: Transport::Http,
            request_body: RequestBody::Json { required: true },
            path_parameters: PathParameterOffsets::default(),
        }
    }

    #[test]
    fn chat_request_is_extracted_without_body_classification() {
        let messages = extract_guardrail_messages(
            operation("openai_chat_completions", "createChatCompletion"),
            GuardrailMessageDirection::Request,
            br#"{"messages":[{"role":"user","content":"hello"}]}"#,
        )
        .unwrap();
        assert_eq!(messages, vec![json!({"role": "user", "content": "hello"})]);
    }

    #[test]
    fn chat_request_rejects_media_for_text_guardrails() {
        let error = extract_guardrail_messages(
            operation("openai_chat_completions", "createChatCompletion"),
            GuardrailMessageDirection::Request,
            br#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.invalid/image.png"}}]}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GuardrailMessageExtractError::UnsupportedContent(_)));
    }

    #[test]
    fn chat_response_rejects_media_for_text_guardrails() {
        let error = extract_guardrail_messages(
            operation("openai_chat_completions", "createChatCompletion"),
            GuardrailMessageDirection::Response,
            br#"{"choices":[{"message":{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"https://example.invalid/image.png"}}]}}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GuardrailMessageExtractError::UnsupportedContent(_)));
    }

    #[test]
    fn responses_request_uses_operation_identity() {
        let messages = extract_guardrail_messages(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Request,
            br#"{"model":"m","input":"hello"}"#,
        )
        .unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }

    #[test]
    fn responses_request_rejects_media_for_text_guardrails() {
        let error = extract_guardrail_messages(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Request,
            br#"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.invalid/image.png"}]}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GuardrailMessageExtractError::UnsupportedContent(_)));
    }

    #[test]
    fn responses_output_text_becomes_assistant_message() {
        let messages = extract_guardrail_messages(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Response,
            br#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}]}"#,
        )
        .unwrap();
        assert_eq!(messages, vec![json!({"role": "assistant", "content": "answer"})]);
    }

    #[test]
    fn responses_output_rejects_media() {
        let error = extract_guardrail_messages(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Response,
            br#"{"output":[{"type":"message","role":"assistant","content":[{"type":"image"}]}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GuardrailMessageExtractError::UnsupportedContent(_)));
    }

    #[test]
    fn anthropic_text_blocks_become_canonical_messages() {
        let messages = extract_guardrail_messages(
            operation("anthropic_messages", "messages_post"),
            GuardrailMessageDirection::Request,
            br#"{"system":"rules","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}"#,
        )
        .unwrap();
        assert_eq!(messages[0], json!({"role": "system", "content": "rules"}));
        assert_eq!(messages[1], json!({"role": "user", "content": "hello"}));
    }

    #[test]
    fn anthropic_request_rejects_tool_content() {
        let error = extract_guardrail_messages(
            operation("anthropic_messages", "messages_post"),
            GuardrailMessageDirection::Request,
            br#"{"messages":[{"role":"user","content":[{"type":"tool_result","content":"secret"}]}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GuardrailMessageExtractError::UnsupportedContent(_)));
    }

    #[test]
    fn anthropic_response_rejects_a_non_assistant_role() {
        let error = extract_guardrail_messages(
            operation("anthropic_messages", "messages_post"),
            GuardrailMessageDirection::Response,
            br#"{"role":"user","content":[{"type":"text","text":"not output"}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GuardrailMessageExtractError::InvalidShape(_)));
    }

    #[test]
    fn unknown_operation_does_not_fallback_to_chat_shape() {
        let error = extract_guardrail_messages(
            operation("openai_responses", "getResponse"),
            GuardrailMessageDirection::Request,
            br#"{"messages":[{"role":"user","content":"hello"}]}"#,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GuardrailMessageExtractError::UnsupportedOperation { .. }
        ));
    }
}
