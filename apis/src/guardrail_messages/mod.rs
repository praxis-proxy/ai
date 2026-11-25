// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-aware extraction of messages for `NeMo` guardrail checks.
//!
//! This module deliberately does not classify request bodies. Callers must
//! provide the request-head identity published by `ai_operation`.

use serde_json::{Map, Value, json};
use thiserror::Error;

mod redaction;
pub use redaction::{GuardrailTextLocations, extract_with_locations, redact_native_body};

use crate::{
    anthropic::routes::is_create_message,
    openai::{
        chat_completions::routes::is_create_chat_completion,
        responses::routes::is_create_response,
        translation::{chat_completions::responses_request_to_chat_messages, reasoning::ReasoningOptions},
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
pub fn extract_guardrail_messages(
    operation: AiOperationMatch,
    direction: GuardrailMessageDirection,
    body: &[u8],
) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let mut json = parse_guardrail_body(operation, direction, body)?;
    extract_parsed_guardrail_messages(operation, direction, &mut json)
}

/// Parse once after validating the trusted operation's transport.
fn parse_guardrail_body(
    operation: AiOperationMatch,
    direction: GuardrailMessageDirection,
    body: &[u8],
) -> Result<Value, GuardrailMessageExtractError> {
    if operation.transport != Transport::Http {
        return Err(GuardrailMessageExtractError::UnsupportedOperation {
            protocol: operation.application_protocol.as_str(),
            operation: operation.operation_id,
            direction,
        });
    }

    serde_json::from_slice(body).map_err(GuardrailMessageExtractError::InvalidJson)
}

/// Borrow native fields so the same document can verify text locations.
/// Chat extraction retains its existing ownership transfer.
fn extract_parsed_guardrail_messages(
    operation: AiOperationMatch,
    direction: GuardrailMessageDirection,
    json: &mut Value,
) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    if is_create_chat_completion(operation.application_protocol, operation.operation_id) {
        return match direction {
            GuardrailMessageDirection::Request => extract_chat_request(std::mem::take(json)),
            GuardrailMessageDirection::Response => extract_chat_response(std::mem::take(json)),
        };
    }
    if is_create_response(operation.application_protocol, operation.operation_id) {
        return match direction {
            GuardrailMessageDirection::Request => extract_responses_request(json),
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

/// Extract text messages without imposing Chat translation restrictions on native Responses options.
fn extract_responses_request(json: &Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let messages = responses_request_to_chat_messages(json, &ReasoningOptions::default())
        .map_err(|_error| GuardrailMessageExtractError::InvalidShape("OpenAI Responses request"))?;
    if messages.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses request input",
        ));
    }
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
fn extract_responses_response(json: &Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let output = json
        .get("output")
        .and_then(Value::as_array)
        .ok_or(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses response output",
        ))?;
    if output.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape(
            "OpenAI Responses response output",
        ));
    }

    let mut messages = Vec::new();
    for item in output {
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
            "reasoning" => validate_responses_reasoning(item)?,
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

/// Skip opaque reasoning state only; visible reasoning needs its own screening policy.
fn validate_responses_reasoning(item: &Value) -> Result<(), GuardrailMessageExtractError> {
    for field in ["summary", "content"] {
        match item.get(field) {
            None | Some(Value::Null) => {},
            Some(Value::Array(parts)) if parts.is_empty() => {},
            Some(Value::Array(_)) => {
                return Err(GuardrailMessageExtractError::UnsupportedContent(
                    "OpenAI Responses reasoning",
                ));
            },
            Some(_) => {
                return Err(GuardrailMessageExtractError::InvalidShape("OpenAI Responses reasoning"));
            },
        }
    }
    Ok(())
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
fn extract_anthropic_request(json: &Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let object = json
        .as_object()
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic request"))?;
    let mut messages = Vec::new();
    if let Some(system) = object.get("system") {
        let content = extract_anthropic_text(system)?;
        if !content.is_empty() {
            messages.push(canonical_text_message("system".to_owned(), content));
        }
    }
    let input = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic request messages"))?;
    if input.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape("Anthropic request messages"));
    }
    for message in input {
        let role = required_string(message, "role", "Anthropic request message role")?;
        let content = message
            .get("content")
            .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic text content"))?;
        messages.push(canonical_text_message(
            role.to_owned(),
            extract_anthropic_text(content)?,
        ));
    }
    Ok(messages)
}

/// Extract the assistant message from an Anthropic Messages response.
fn extract_anthropic_response(json: &Value) -> Result<Vec<Value>, GuardrailMessageExtractError> {
    let object = json
        .as_object()
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic response"))?;
    let role = required_string(json, "role", "Anthropic response role")?;
    if role != "assistant" {
        return Err(GuardrailMessageExtractError::InvalidShape("Anthropic response role"));
    }
    let content = object
        .get("content")
        .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic text content"))?;
    let content = extract_anthropic_text(content)?;
    if content.is_empty() {
        return Err(GuardrailMessageExtractError::InvalidShape("Anthropic response content"));
    }
    Ok(vec![canonical_text_message(role.to_owned(), content)])
}

/// Extract the textual content of an Anthropic Messages value.
/// Canonical messages own their text while the native fields remain available
/// for location verification, so string content is copied at this boundary.
fn extract_anthropic_text(value: &Value) -> Result<String, GuardrailMessageExtractError> {
    let blocks = match value {
        Value::String(text) => return Ok(text.to_owned()),
        Value::Array(blocks) => blocks,
        _ => return Err(GuardrailMessageExtractError::InvalidShape("Anthropic text content")),
    };
    let mut text = String::new();
    for block in blocks {
        let block = block
            .as_object()
            .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic content block"))?;
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or(GuardrailMessageExtractError::InvalidShape("Anthropic text block"))?,
            ),
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

    /// Native extraction must leave the parsed document intact for path verification.
    fn assert_native_document_preserved(
        operation: AiOperationMatch,
        direction: GuardrailMessageDirection,
        body: &Value,
        expected: &[Value],
    ) {
        let bytes = serde_json::to_vec(body).unwrap();
        let mut parsed = parse_guardrail_body(operation, direction, &bytes).unwrap();
        let messages = extract_parsed_guardrail_messages(operation, direction, &mut parsed).unwrap();
        assert_eq!(messages.as_slice(), expected);
        assert_eq!(&parsed, body, "location checks must see the original native fields");
        let (located, _) = extract_with_locations(operation, direction, &bytes).unwrap();
        assert_eq!(located.as_slice(), expected);
    }

    #[test]
    fn parsed_responses_request_preserves_cached_text_and_native_fields() {
        assert_native_document_preserved(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Request,
            &json!({"instructions":"rules", "input":[{"role":"user","content":[{
                "type":"input_text","text":"café","prompt_cache_breakpoint":{"mode":"explicit"}
            }]}]}),
            &[
                json!({"role":"system","content":"rules"}),
                json!({"role":"user","content":[{
                    "type":"text","text":"café","prompt_cache_breakpoint":{"mode":"explicit"}
                }]}),
            ],
        );
    }

    #[test]
    fn parsed_responses_response_preserves_empty_output_refusal_and_reasoning() {
        assert_native_document_preserved(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Response,
            &json!({"output":[
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":""}]},
                {"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"refused"}]},
                {"type":"reasoning","encrypted_content":"opaque"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
            ]}),
            &[
                json!({"role":"assistant","content":"refused"}),
                json!({"role":"assistant","content":"answer"}),
            ],
        );
    }

    #[test]
    fn parsed_anthropic_request_preserves_strings_and_multiple_blocks() {
        assert_native_document_preserved(
            operation("anthropic_messages", "messages_post"),
            GuardrailMessageDirection::Request,
            &json!({"system":[{"type":"text","text":"rules"}], "messages":[
                {"role":"user","content":"first"},
                {"role":"assistant","content":[{"type":"text","text":"reply"}]},
                {"role":"user","content":[{"type":"text","text":"sec"},{"type":"text","text":"ond"}]}
            ]}),
            &[
                json!({"role":"system","content":"rules"}),
                json!({"role":"user","content":"first"}),
                json!({"role":"assistant","content":"reply"}),
                json!({"role":"user","content":"second"}),
            ],
        );
    }

    #[test]
    fn parsed_anthropic_response_preserves_text_metadata() {
        assert_native_document_preserved(
            operation("anthropic_messages", "messages_post"),
            GuardrailMessageDirection::Response,
            &json!({"role":"assistant","content":[{"type":"text","text":"café","citations":null}],
                "stop_details":{"type":"refusal","category":"general_harms","explanation":"details"}}),
            &[json!({"role":"assistant","content":"café"})],
        );
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
    fn responses_request_extracts_messages_independently_of_native_options() {
        let tools = json!([{"type":"code_interpreter","container":{"type":"auto"}}]);
        for (field, value) in [
            ("truncation", json!("auto")),
            ("background", json!(true)),
            ("tools", tools),
            ("tool_choice", json!({"type":"code_interpreter"})),
        ] {
            let mut request = json!({
                "model":"m", "instructions":"Follow policy",
                "input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}]
            });
            request[field] = value;
            let body = serde_json::to_vec(&request).unwrap();
            let messages = extract_guardrail_messages(
                operation("openai_responses", "createResponse"),
                GuardrailMessageDirection::Request,
                &body,
            )
            .unwrap();
            assert_eq!(
                messages,
                vec![
                    json!({"role":"system","content":"Follow policy"}),
                    json!({"role":"user","content":"hello"})
                ],
                "{field} must not prevent canonical conversation extraction"
            );
        }
    }

    #[test]
    fn responses_request_rejects_invalid_input_even_with_native_options() {
        for input in [json!(17), json!(true), json!([]), json!([{"role":"user","content":17}])] {
            let body = serde_json::to_vec(&json!({"input":input,"truncation":"auto"})).unwrap();
            let error = extract_guardrail_messages(
                operation("openai_responses", "createResponse"),
                GuardrailMessageDirection::Request,
                &body,
            )
            .unwrap_err();
            assert!(
                matches!(error, GuardrailMessageExtractError::InvalidShape(_)),
                "native inference options must not bypass input validation"
            );
        }
    }

    #[test]
    fn responses_request_rejects_media_for_text_guardrails() {
        let error = extract_guardrail_messages(
            operation("openai_responses", "createResponse"),
            GuardrailMessageDirection::Request,
            br#"{"model":"m","truncation":"auto","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.invalid/image.png"}]}]}"#,
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
    fn responses_output_rejects_visible_reasoning() {
        for reasoning in [
            json!({"type":"reasoning","summary":[{"type":"summary_text","text":"secret summary"}]}),
            json!({"type":"reasoning","content":[{"type":"reasoning_text","text":"secret reasoning"}]}),
            json!({"type":"reasoning","summary":[{"type":"unknown","text":"secret summary"}]}),
        ] {
            for output in [
                json!([reasoning, {"type":"message","role":"assistant","content":[{"type":"output_text","text":"safe"}]}]),
                json!([{"type":"message","role":"assistant","content":[{"type":"output_text","text":"safe"}]}, reasoning]),
            ] {
                let body = serde_json::to_vec(&json!({"output":output})).unwrap();
                let error = extract_guardrail_messages(
                    operation("openai_responses", "createResponse"),
                    GuardrailMessageDirection::Response,
                    &body,
                )
                .unwrap_err();
                assert!(
                    matches!(
                        error,
                        GuardrailMessageExtractError::UnsupportedContent("OpenAI Responses reasoning")
                    ),
                    "visible reasoning must reject the entire response regardless of its output position"
                );
            }
        }
    }

    #[test]
    fn responses_output_rejects_malformed_reasoning_fields() {
        for reasoning in [
            json!({"type":"reasoning","summary":"secret summary"}),
            json!({"type":"reasoning","content":{"text":"secret reasoning"}}),
        ] {
            let body = serde_json::to_vec(&json!({"output":[reasoning]})).unwrap();
            let error = extract_guardrail_messages(
                operation("openai_responses", "createResponse"),
                GuardrailMessageDirection::Response,
                &body,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    GuardrailMessageExtractError::InvalidShape("OpenAI Responses reasoning")
                ),
                "malformed reasoning must not be silently skipped"
            );
        }
    }

    #[test]
    fn responses_output_allows_opaque_reasoning_state_with_text_output() {
        for reasoning in [
            json!({"type":"reasoning","encrypted_content":"opaque"}),
            json!({"type":"reasoning","summary":[],"content":[],"encrypted_content":"opaque"}),
        ] {
            let body = serde_json::to_vec(&json!({"output":[
                reasoning,
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}
            ]}))
            .unwrap();
            let messages = extract_guardrail_messages(
                operation("openai_responses", "createResponse"),
                GuardrailMessageDirection::Response,
                &body,
            )
            .unwrap();
            assert_eq!(messages, vec![json!({"role":"assistant","content":"answer"})]);
        }
    }

    #[test]
    fn anthropic_response_rejects_thinking_alongside_text() {
        for thinking in [
            json!({"type":"thinking","thinking":"secret reasoning","signature":"opaque"}),
            json!({"type":"redacted_thinking","data":"opaque"}),
        ] {
            let body = serde_json::to_vec(&json!({"role":"assistant","content":[
                thinking, {"type":"text","text":"answer"}
            ]}))
            .unwrap();
            let error = extract_guardrail_messages(
                operation("anthropic_messages", "messages_post"),
                GuardrailMessageDirection::Response,
                &body,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    GuardrailMessageExtractError::UnsupportedContent("Anthropic message")
                ),
                "Anthropic thinking blocks must reject the response even when normal text is present"
            );
        }
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
