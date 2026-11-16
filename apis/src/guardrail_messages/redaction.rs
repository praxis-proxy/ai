// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Native text locations used to apply guardrail modifications.

use serde_json::Value;

use super::{
    GuardrailMessageDirection, GuardrailMessageExtractError, extract_parsed_guardrail_messages, parse_guardrail_body,
};
use crate::{
    anthropic::routes::is_create_message, openai::responses::routes::is_create_response,
    operation_classifier::AiOperationMatch,
};

/// Original native locations indexed by the corresponding checked message.
#[derive(Debug)]
pub struct GuardrailTextLocations {
    /// Native JSON paths indexed by canonical message index.
    paths: Vec<Option<String>>,
    /// Whether the operation uses the Responses protocol.
    is_responses: bool,
    /// Direction of the checked body.
    direction: GuardrailMessageDirection,
}

impl GuardrailTextLocations {
    /// Whether a checked message is a native Responses refusal.
    pub fn is_refusal(&self, index: usize) -> bool {
        self.paths
            .get(index)
            .and_then(Option::as_deref)
            .is_some_and(|path| path.ends_with("/refusal"))
    }
}

/// Extract the existing canonical messages and record their native text locations.
///
/// Tool-answer reordering preserves the order of user messages, including
/// orphan tool outputs translated to user messages. Derived messages are
/// deliberately not rewritable.
///
/// # Errors
/// Returns an error if extraction or native location verification fails.
#[expect(
    clippy::too_many_lines,
    reason = "validates native locations against canonical messages"
)]
pub fn extract_with_locations(
    operation: AiOperationMatch,
    direction: GuardrailMessageDirection,
    body: &[u8],
) -> Result<(Vec<Value>, GuardrailTextLocations), GuardrailMessageExtractError> {
    let mut json = parse_guardrail_body(operation, direction, body)?;
    let messages = extract_parsed_guardrail_messages(operation, direction, &mut json)?;
    let responses = is_create_response(operation.application_protocol, operation.operation_id);
    let anthropic = is_create_message(operation.application_protocol, operation.operation_id);
    let mut paths = vec![None; messages.len()];
    if responses || anthropic {
        let candidates = native_candidates(&json, direction, responses);
        let role = match direction {
            GuardrailMessageDirection::Request => "user",
            GuardrailMessageDirection::Response => "assistant",
        };
        let mut candidates = candidates.into_iter();
        for (index, message) in messages.iter().enumerate() {
            if message.get("role").and_then(Value::as_str) != Some(role) {
                continue;
            }
            let path = candidates
                .next()
                .ok_or(GuardrailMessageExtractError::InvalidShape("guardrail text locations"))?;
            if let Some(path) = &path {
                let text = json.pointer(path).and_then(Value::as_str);
                if text != canonical_text(message) {
                    return Err(GuardrailMessageExtractError::InvalidShape(
                        "guardrail text location mismatch",
                    ));
                }
            }
            let slot = paths
                .get_mut(index)
                .ok_or(GuardrailMessageExtractError::InvalidShape("guardrail text locations"))?;
            *slot = path;
        }
        if candidates.next().is_some() {
            return Err(GuardrailMessageExtractError::InvalidShape("guardrail text locations"));
        }
    }
    Ok((
        messages,
        GuardrailTextLocations {
            paths,
            is_responses: responses,
            direction,
        },
    ))
}

/// Return text from a canonical message, including a single cached text part.
fn canonical_text(message: &Value) -> Option<&str> {
    match message.get("content")? {
        Value::String(text) => Some(text.as_str()),
        Value::Array(parts) if parts.len() == 1 => parts.first()?.get("text")?.as_str(),
        _ => None,
    }
}

/// Identify native targets in their original order. A `None` entry represents
/// a selected message that cannot be rewritten, such as a multi-block message.
#[expect(
    clippy::too_many_lines,
    reason = "dispatches all native operation and direction combinations"
)]
fn native_candidates(json: &Value, direction: GuardrailMessageDirection, responses: bool) -> Vec<Option<String>> {
    match (responses, direction) {
        (true, GuardrailMessageDirection::Request) => match json.get("input") {
            Some(Value::String(_)) => vec![Some("/input".to_owned())],
            Some(Value::Array(items)) => items
                .iter()
                .enumerate()
                .filter(|(_, item)| responses_request_target(item))
                .map(|(i, item)| responses_request_candidate(item, &format!("/input/{i}/content")))
                .collect(),
            Some(item @ Value::Object(_)) if responses_request_target(item) => {
                vec![responses_request_candidate(item, "/input/content")]
            },
            _ => vec![],
        },
        (true, GuardrailMessageDirection::Response) => json
            .get("output")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .filter_map(|(i, item)| {
                        match item.get("type").and_then(Value::as_str) {
                            Some("function_call") => Some(None),
                            Some("message") => {
                                let parts = item.get("content").and_then(Value::as_array)?;
                                // The extractor omits messages with empty concatenated text.
                                let nonempty = parts.iter().any(|part| {
                                    part.get("text")
                                        .or_else(|| part.get("refusal"))
                                        .and_then(Value::as_str)
                                        .is_some_and(|s| !s.is_empty())
                                });
                                nonempty
                                    .then(|| content_path(item.get("content"), &format!("/output/{i}/content"), false))
                            },
                            _ => None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
        (false, GuardrailMessageDirection::Request) => json
            .get("messages")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| item.get("role").and_then(Value::as_str) == Some("user"))
                    .map(|(i, item)| content_path(item.get("content"), &format!("/messages/{i}/content"), true))
                    .collect()
            })
            .unwrap_or_default(),
        (false, GuardrailMessageDirection::Response) => vec![content_path(json.get("content"), "/content", false)],
    }
}

/// Identify user messages, including orphan tool outputs lowered to users.
fn responses_request_target(item: &Value) -> bool {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call_output") => item.get("call_id").is_none_or(Value::is_null),
        None | Some("message") => item.get("role").and_then(Value::as_str) == Some("user"),
        Some(_) => false,
    }
}

/// Keep orphan tool outputs evaluable but nonrewritable.
fn responses_request_candidate(item: &Value, path: &str) -> Option<String> {
    if item.get("type").and_then(Value::as_str) == Some("function_call_output") {
        None
    } else {
        content_path(item.get("content"), path, true)
    }
}

/// Resolve one native string or one supported text block to its JSON path.
fn content_path(content: Option<&Value>, base: &str, allow_string: bool) -> Option<String> {
    match content? {
        Value::String(_) if allow_string => Some(base.to_owned()),
        Value::Array(parts) if parts.len() == 1 => {
            let part = parts.first()?;
            let field = match part.get("type").and_then(Value::as_str)? {
                "text" | "input_text" | "output_text" => "text",
                "refusal" => "refusal",
                _ => return None,
            };
            part.get(field)?.as_str()?;
            Some(format!("{base}/0/{field}"))
        },
        _ => None,
    }
}

/// Apply all replacements to a private native document and return it.
///
/// # Errors
/// Returns an error for an unsupported or ambiguous location or unsafe fields.
#[expect(clippy::too_many_lines, reason = "validates and applies a complete native rewrite")]
pub fn redact_native_body(
    body: &[u8],
    locations: &GuardrailTextLocations,
    replacements: impl IntoIterator<Item = (usize, String)>,
) -> Result<Value, &'static str> {
    let mut json: Value = serde_json::from_slice(body).map_err(|_error| "invalid JSON body")?;
    if !locations.is_responses && locations.direction == GuardrailMessageDirection::Response {
        validate_anthropic_response_fields(&json)?;
        clear_anthropic_refusal_explanation(&mut json)?;
    }
    if locations.is_responses
        && locations.direction == GuardrailMessageDirection::Response
        && json.get("output_text").is_some_and(|value| !value.is_null())
    {
        return Err("response contains an aggregate text alias");
    }
    if locations.is_responses
        && locations.direction == GuardrailMessageDirection::Response
        && json.get("output").and_then(Value::as_array).is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        })
    {
        return Err("response contains a function call");
    }
    let mut seen = std::collections::HashSet::new();
    let mut count = 0;
    for (index, replacement) in replacements {
        count += 1;
        let path = locations
            .paths
            .get(index)
            .and_then(Option::as_deref)
            .ok_or("unsupported text location")?;
        if !seen.insert(path.to_owned()) {
            return Err("duplicate text location");
        }
        validate_message_fields(&json, locations, path)?;
        if replacement.is_empty()
            && locations.direction == GuardrailMessageDirection::Request
            && !locations.is_responses
        {
            return Err("empty Anthropic request text cannot be safely forwarded");
        }
        clear_text_metadata(&mut json, locations, path)?;
        let target = json.pointer_mut(path).ok_or("missing text location")?;
        if !target.is_string() {
            return Err("invalid text location");
        }
        *target = Value::String(replacement);
    }
    if count == 0 {
        return Err("no replacements");
    }
    if locations.is_responses && locations.direction == GuardrailMessageDirection::Response {
        validate_responses_sibling_metadata(&json)?;
    }
    Ok(json)
}

/// Rewritten blocks have already had their metadata cleared. Any remaining
/// annotations or log probabilities belong to content that was not rewritten,
/// including empty-text outputs omitted by extraction. Refuse the whole
/// rewrite rather than forwarding or silently removing that unscreened data.
fn validate_responses_sibling_metadata(json: &Value) -> Result<(), &'static str> {
    let output = json
        .get("output")
        .and_then(Value::as_array)
        .ok_or("invalid Responses output")?;
    for message in output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
    {
        let blocks = message
            .get("content")
            .and_then(Value::as_array)
            .ok_or("invalid Responses message content")?;
        for block in blocks {
            for field in ["annotations", "logprobs"] {
                match block.get(field) {
                    None | Some(Value::Null) => {},
                    Some(Value::Array(values)) if values.is_empty() => {},
                    Some(_) => return Err("unmodified response block contains unscreened metadata"),
                }
            }
        }
    }
    Ok(())
}

/// Reject unknown fields on a response that will be rewritten.
fn validate_anthropic_response_fields(json: &Value) -> Result<(), &'static str> {
    let message = json.as_object().ok_or("invalid Anthropic response")?;
    if message.keys().any(|key| {
        ![
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "stop_details",
            "usage",
            "container",
        ]
        .contains(&key.as_str())
    }) {
        return Err("unknown Anthropic response field");
    }
    Ok(())
}

/// Clear refusal text that could repeat content masked by the guardrail.
fn clear_anthropic_refusal_explanation(json: &mut Value) -> Result<(), &'static str> {
    let Some(details) = json.get_mut("stop_details").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let details = details.as_object_mut().ok_or("invalid refusal details")?;
    if details
        .keys()
        .any(|key| !["type", "category", "explanation"].contains(&key.as_str()))
        || details.get("type").and_then(Value::as_str) != Some("refusal")
        || !matches!(details.get("category"), Some(Value::Null))
            && !details.get("category").and_then(Value::as_str).is_some_and(|category| {
                ["cyber", "bio", "frontier_llm", "reasoning_extraction", "general_harms"].contains(&category)
            })
    {
        return Err("invalid refusal details");
    }
    let explanation = details.get_mut("explanation").ok_or("invalid refusal explanation")?;
    if !explanation.is_null() && !explanation.is_string() {
        return Err("invalid refusal explanation");
    }
    *explanation = Value::Null;
    Ok(())
}

/// Reject unknown fields on a message whose text will be changed.
fn validate_message_fields(json: &Value, locations: &GuardrailTextLocations, path: &str) -> Result<(), &'static str> {
    if locations.direction == GuardrailMessageDirection::Request && path.contains("/content") {
        let message_path = path.split("/content").next().ok_or("invalid input location")?;
        let message = json
            .pointer(message_path)
            .and_then(Value::as_object)
            .ok_or("invalid input message")?;
        let allowed: &[&str] = if locations.is_responses {
            &["role", "content", "type", "id", "status", "phase"]
        } else {
            &["role", "content"]
        };
        if message.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err("unknown input message field");
        }
    }
    if locations.is_responses && locations.direction == GuardrailMessageDirection::Response {
        let message_path = path.split("/content/").next().ok_or("invalid output location")?;
        let message = json
            .pointer(message_path)
            .and_then(Value::as_object)
            .ok_or("invalid output message")?;
        if message
            .keys()
            .any(|key| !["id", "type", "role", "status", "content", "phase"].contains(&key.as_str()))
        {
            return Err("unknown output message field");
        }
    }
    Ok(())
}

/// Remove text-bearing metadata from a rewritten native text block.
#[expect(clippy::too_many_lines, reason = "validates and cleans each native text block")]
fn clear_text_metadata(json: &mut Value, locations: &GuardrailTextLocations, path: &str) -> Result<(), &'static str> {
    if let Some((parent, field)) = path.rsplit_once('/')
        && (field == "text" || field == "refusal")
    {
        let block = json
            .pointer_mut(parent)
            .and_then(Value::as_object_mut)
            .ok_or("invalid text block")?;
        let allowed: &[&str] = if locations.is_responses && field == "refusal" {
            &["type", "refusal", "annotations", "logprobs", "prompt_cache_breakpoint"]
        } else if locations.is_responses {
            &["type", "text", "annotations", "logprobs", "prompt_cache_breakpoint"]
        } else {
            &["type", "text", "citations", "cache_control"]
        };
        if block.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err("unknown text block field");
        }
        if locations.is_responses {
            if block.contains_key("annotations") {
                block.insert("annotations".into(), Value::Array(vec![]));
            }
            if block.contains_key("logprobs") {
                block.insert("logprobs".into(), Value::Array(vec![]));
            }
        } else if locations.direction == GuardrailMessageDirection::Response {
            block.insert("citations".into(), Value::Null);
        } else {
            block.remove("citations");
        }
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
#[expect(clippy::indexing_slicing, reason = "tests assert known message indices")]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::operation::{ApplicationProtocol, PathParameterOffsets, RequestBody, Transport};

    fn responses() -> AiOperationMatch {
        AiOperationMatch {
            application_protocol: ApplicationProtocol::new("openai_responses"),
            operation_id: "createResponse",
            transport: Transport::Http,
            request_body: RequestBody::Json { required: true },
            path_parameters: PathParameterOffsets::default(),
        }
    }

    fn anthropic() -> AiOperationMatch {
        AiOperationMatch {
            application_protocol: ApplicationProtocol::new("anthropic_messages"),
            operation_id: "messages_post",
            transport: Transport::Http,
            request_body: RequestBody::Json { required: true },
            path_parameters: PathParameterOffsets::default(),
        }
    }

    /// Verify rejection also when an empty sibling shifts canonical indices.
    fn assert_response_sibling_metadata_rejected(content: &Value) {
        let body = json!({"output":[
            {"type":"message","role":"assistant","content":content},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"secret"}]}
        ]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (messages, locations) =
            extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
        let index = messages
            .iter()
            .position(|message| message["content"] == "secret")
            .unwrap();
        assert_eq!(
            redact_native_body(&bytes, &locations, [(index, "masked".into())]).unwrap_err(),
            "unmodified response block contains unscreened metadata"
        );
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), body);
    }

    #[test]
    fn responses_redaction_rejects_passed_and_empty_sibling_metadata() {
        for (field, metadata) in [
            (
                "annotations",
                json!([{"type":"url_citation","title":"secret","url":"https://example.test","start_index":0,"end_index":0}]),
            ),
            (
                "logprobs",
                json!([{"token":"secret","bytes":[115,101,99,114,101,116],"logprob":-0.1,"top_logprobs":[]}]),
            ),
            ("annotations", json!("secret")),
            ("logprobs", json!({})),
        ] {
            for text in ["safe", ""] {
                let mut sibling = json!({"type":"output_text","text":text});
                sibling[field] = metadata.clone();
                assert_response_sibling_metadata_rejected(&json!([sibling]));
                assert_response_sibling_metadata_rejected(&json!([{"type":"output_text","text":"safe"}, sibling]));
            }
        }
    }

    #[test]
    fn responses_redaction_preserves_empty_sibling_metadata() {
        for text in ["safe", ""] {
            for metadata in [None, Some(Value::Null), Some(json!([]))] {
                let mut sibling = json!({"type":"output_text","text":text});
                if let Some(metadata) = metadata {
                    sibling["annotations"] = metadata.clone();
                    sibling["logprobs"] = metadata;
                }
                let body = json!({"output":[
                    {"type":"message","role":"assistant","content":[sibling]},
                    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"secret","annotations":[{"title":"secret"}]}]}
                ]});
                let bytes = serde_json::to_vec(&body).unwrap();
                let (messages, locations) =
                    extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
                let index = messages
                    .iter()
                    .position(|message| message["content"] == "secret")
                    .unwrap();
                let result = redact_native_body(&bytes, &locations, [(index, "masked".into())]).unwrap();
                assert_eq!(result.pointer("/output/0"), body.pointer("/output/0"));
                assert_eq!(result.pointer("/output/1/content/0/text"), Some(&json!("masked")));
                assert_eq!(result.pointer("/output/1/content/0/annotations"), Some(&json!([])));
            }
        }
    }

    #[test]
    fn responses_redaction_checks_sibling_metadata_after_all_replacements() {
        let body = json!({"output":[
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"first secret","annotations":[{"title":"secret"}]}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"second secret","logprobs":[{"token":"secret"}]}]}
        ]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
        for replacements in [
            [(0, "first masked".to_owned()), (1, "second masked".to_owned())],
            [(1, "second masked".to_owned()), (0, "first masked".to_owned())],
        ] {
            let result = redact_native_body(&bytes, &locations, replacements).unwrap();
            assert_eq!(result.pointer("/output/0/content/0/text"), Some(&json!("first masked")));
            assert_eq!(
                result.pointer("/output/1/content/0/text"),
                Some(&json!("second masked"))
            );
            assert_eq!(result.pointer("/output/0/content/0/annotations"), Some(&json!([])));
            assert_eq!(result.pointer("/output/1/content/0/logprobs"), Some(&json!([])));
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks mapping and rewrite safety across orphan output shapes"
    )]
    fn orphan_tool_outputs_remain_evaluable_and_cannot_be_rewritten() {
        for orphan in [
            json!({"type":"function_call_output","output":"same"}),
            json!({"type":"function_call_output","call_id":null,"output":"same"}),
            json!({"type":"function_call_output","output":[{"type":"input_text","text":"same"}]}),
        ] {
            let body = json!({"input":[
                {"type":"function_call","call_id":"call1","name":"tool","arguments":"{}"},
                orphan,
                {"role":"user","content":"same"},
                {"type":"function_call_output","call_id":"call1","output":"linked"}
            ]});
            let bytes = serde_json::to_vec(&body).unwrap();
            let (messages, locations) =
                extract_with_locations(responses(), GuardrailMessageDirection::Request, &bytes).unwrap();
            let users: Vec<_> = messages
                .iter()
                .enumerate()
                .filter(|(_, message)| message["role"] == "user")
                .collect();
            assert_eq!(users.len(), 2, "both orphan output and user message must be screened");
            assert!(
                redact_native_body(&bytes, &locations, [(users[0].0, "masked".into())]).is_err(),
                "an orphan tool output cannot be rewritten as user content"
            );
            let rewritten = redact_native_body(&bytes, &locations, [(users[1].0, "masked".into())]).unwrap();
            assert_eq!(
                rewritten.pointer("/input/1"),
                body.pointer("/input/1"),
                "orphan output must stay unchanged"
            );
            assert_eq!(
                rewritten.pointer("/input/2/content"),
                Some(&json!("masked")),
                "only the user message must change"
            );
        }
    }

    #[test]
    fn anthropic_response_redaction_clears_refusal_explanation_and_keeps_citations() {
        let body = json!({"role":"assistant","content":[{"type":"text","text":"secret","citations":null}],
            "stop_details":{"type":"refusal","category":"general_harms","explanation":"secret"}});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(anthropic(), GuardrailMessageDirection::Response, &bytes).unwrap();
        let result = redact_native_body(&bytes, &locations, [(0, "masked".into())]).unwrap();
        assert_eq!(
            result.pointer("/content/0/text"),
            Some(&json!("masked")),
            "response text must be masked"
        );
        assert_eq!(
            result.pointer("/content/0/citations"),
            Some(&Value::Null),
            "response citations must remain present"
        );
        assert_eq!(
            result.pointer("/stop_details/explanation"),
            Some(&Value::Null),
            "refusal explanation must not leak text"
        );
        assert_eq!(
            result.pointer("/stop_details/category"),
            Some(&json!("general_harms")),
            "refusal category must remain"
        );
    }

    #[test]
    fn anthropic_response_unknown_refusal_fields_fail_closed() {
        let body = json!({"role":"assistant","content":[{"type":"text","text":"secret"}],
            "stop_details":{"type":"refusal","category":"general_harms","explanation":null,"extra":"secret"}});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(anthropic(), GuardrailMessageDirection::Response, &bytes).unwrap();
        assert!(
            redact_native_body(&bytes, &locations, [(0, "masked".into())]).is_err(),
            "unknown refusal metadata must not escape a rewrite"
        );
    }

    #[test]
    fn anthropic_response_unknown_message_fields_fail_closed() {
        let body = json!({"role":"assistant","content":[{"type":"text","text":"secret"}],"extra":"secret"});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(anthropic(), GuardrailMessageDirection::Response, &bytes).unwrap();
        assert!(
            redact_native_body(&bytes, &locations, [(0, "masked".into())]).is_err(),
            "unknown response metadata must not escape a rewrite"
        );
    }

    #[test]
    fn responses_response_redaction_rejects_aggregate_text_alias() {
        for alias in [json!("secret"), json!(""), json!([])] {
            let body = json!({"output_text":alias,"output":[
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"secret"}]}
            ]});
            let bytes = serde_json::to_vec(&body).unwrap();
            let (messages, locations) =
                extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
            assert_eq!(
                messages,
                vec![json!({"role":"assistant","content":"secret"})],
                "the fixture must reach rewriting after successful text extraction"
            );
            assert_eq!(
                redact_native_body(&bytes, &locations, [(0, "masked".into())]).unwrap_err(),
                "response contains an aggregate text alias",
                "a parallel text alias must reject the whole rewrite"
            );
        }
        let body = json!({"output_text":null,"output":[
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"secret"}]}
        ]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
        let mut expected = body;
        expected["output"][0]["content"][0]["text"] = json!("masked");
        assert_eq!(
            redact_native_body(&bytes, &locations, [(0, "masked".into())]).unwrap(),
            expected,
            "a null alias contains no duplicate text and must remain untouched"
        );
    }

    #[test]
    fn responses_response_redaction_rejects_function_call_siblings() {
        for call_first in [false, true] {
            let text = json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"secret"}]});
            let call = json!({"type":"function_call","call_id":"call_one","name":"example_tool","arguments":"{\"sensitive\":\"secret\"}"});
            let output = if call_first { vec![call, text] } else { vec![text, call] };
            let bytes = serde_json::to_vec(&json!({"output":output})).unwrap();
            let (messages, locations) =
                extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
            assert_eq!(messages.len(), 2, "both native outputs must pass extraction");
            let index = messages
                .iter()
                .position(|message| message["content"] == "secret")
                .unwrap();
            assert_eq!(
                redact_native_body(&bytes, &locations, [(index, "masked".into())]).unwrap_err(),
                "response contains a function call",
                "masking text must not forward unscreened sibling tool arguments"
            );
        }
    }

    #[test]
    fn responses_blocks_reject_secondary_text_fields() {
        for block in [
            json!({"type":"output_text","text":"secret","refusal":"secret","annotations":[]}),
            json!({"type":"refusal","refusal":"secret","text":"secret"}),
        ] {
            let body = json!({"output":[{"type":"message","role":"assistant","content":[block]}]});
            let bytes = serde_json::to_vec(&body).unwrap();
            let (_, locations) =
                extract_with_locations(responses(), GuardrailMessageDirection::Response, &bytes).unwrap();
            assert!(
                redact_native_body(&bytes, &locations, [(0, "masked".into())]).is_err(),
                "a secondary text field must not escape a rewrite"
            );
        }
    }

    #[test]
    fn user_locations_survive_tool_answer_reordering() {
        let body = json!({"instructions":"rules", "input":[
            {"type":"message","role":"user","content":"first"},
            {"type":"function_call","call_id":"call1","name":"tool","arguments":"{}"},
            {"type":"message","role":"user","content":"second"},
            {"type":"function_call_output","call_id":"call1","output":"done"}
        ]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (messages, locations) =
            extract_with_locations(responses(), GuardrailMessageDirection::Request, &bytes).unwrap();
        let indices: Vec<_> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.get("role").and_then(Value::as_str) == Some("user"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(indices, vec![1, 4]);
        let result = redact_native_body(
            &bytes,
            &locations,
            [(indices[1], "two".into()), (indices[0], "one".into())],
        )
        .unwrap();
        assert_eq!(result.pointer("/input/0/content"), Some(&json!("one")));
        assert_eq!(result.pointer("/input/2/content"), Some(&json!("two")));
        assert_eq!(result.pointer("/input/3/output"), Some(&json!("done")));
    }

    #[test]
    fn ambiguous_or_duplicate_replacements_reject_the_entire_rewrite() {
        let body = json!({"input":[
            {"role":"user","content":"first"},
            {"role":"user","content":[{"type":"input_text","text":"sec"},{"type":"input_text","text":"ond"}]}
        ]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(responses(), GuardrailMessageDirection::Request, &bytes).unwrap();
        assert!(redact_native_body(&bytes, &locations, [(0, "one".into()), (1, "two".into())]).is_err());
        assert!(redact_native_body(&bytes, &locations, [(0, "one".into()), (0, "two".into())]).is_err());
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), body);
    }

    #[test]
    fn unknown_fields_on_the_target_message_are_not_forwarded_with_a_redaction() {
        let body = json!({"input":[{"role":"user","content":"secret","extra":"secret"}]});
        let bytes = serde_json::to_vec(&body).unwrap();
        let (_, locations) = extract_with_locations(responses(), GuardrailMessageDirection::Request, &bytes).unwrap();
        assert!(redact_native_body(&bytes, &locations, [(0, "masked".into())]).is_err());
    }
}
