// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Import request-hashed provider recordings into the shared fixture schema.

#![allow(
    clippy::missing_docs_in_private_items,
    reason = "private serde DTOs mirror the external recording envelope"
)]

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use super::{
    FixtureError, RecordedBody, RecordedExchange, RecordedRequest, RecordedResponse, SseFrame,
    header_policy::is_credential_header,
};

/// Maximum recursive descent while removing external response wrappers.
const MAX_EXTERNAL_RESPONSE_DEPTH: usize = 128;

/// An upstream exchange imported from an external request-hashed recording.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedUpstream {
    /// Stable source-test provenance, if the external recording supplied it.
    pub source_id: Option<String>,
    /// Provider provenance when the source explicitly supplies a provider name.
    pub provider: Option<String>,
    /// The upstream model recorded on the request.
    pub model: Option<String>,
    /// The normalized request and upstream response exchange.
    pub exchange: RecordedExchange,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalEnvelope {
    test_id: String,
    request: ExternalRequest,
    response: ExternalResponse,
    #[serde(default)]
    id_normalization_mapping: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalRequest {
    method: String,
    endpoint: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, ExternalHeaderValues>,
    model: String,
    body: Value,
    #[serde(default)]
    provider_metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExternalHeaderValues {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalResponse {
    body: Value,
    is_streaming: bool,
    #[serde(default = "default_response_status")]
    status: u16,
}

const fn default_response_status() -> u16 {
    200
}

/// Import an external request-hashed provider recording into a shared exchange.
///
/// The importer accepts only the documented envelope and removes wrappers that
/// have exactly `__type__` and `__data__` keys. It intentionally treats source
/// test ids as provenance only; source repository names are not inferred.
///
/// # Errors
///
/// Returns an error when the envelope is malformed, a wrapper is ambiguous, a
/// request body is absent, or the response body conflicts with `is_streaming`.
/// Errors deliberately omit external body and header values.
#[expect(
    clippy::too_many_lines,
    reason = "the envelope conversion is kept together so every boundary validation is visible"
)]
pub fn import_external_recording(content: &str) -> Result<ImportedUpstream, FixtureError> {
    let ExternalEnvelope {
        test_id,
        request,
        response,
        id_normalization_mapping,
    } = serde_json::from_str(content).map_err(FixtureError::ExternalRecordingJson)?;

    let ExternalRequest {
        method,
        endpoint,
        url,
        headers,
        model,
        body: request_body,
        provider_metadata,
    } = request;

    // These source-only fields are parsed to keep the external envelope strict,
    // but do not become permanent fixture provenance or routing metadata.
    let _ = (id_normalization_mapping, url, provider_metadata);

    if request_body.is_null() {
        return Err(external_error("request body is required"));
    }
    validate_response_status(response.status)?;

    let response_content_type = if response.is_streaming {
        "text/event-stream"
    } else {
        "application/json"
    };
    let response_body = unwrap_typed_wrappers(response.body)?;
    let response_body = if response.is_streaming {
        let Value::Array(chunks) = response_body else {
            return Err(external_error("streaming response body must be an array"));
        };
        if chunks.is_empty() {
            return Err(external_error("streaming response body must be a nonempty array"));
        }
        let mut frames = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            if !chunk.is_object() {
                return Err(external_error("streaming response chunks must be objects"));
            }
            let data = serde_json::to_string(&chunk).map_err(FixtureError::JsonBodyRender)?;
            frames.push(SseFrame {
                event: None,
                data,
                id: None,
                retry: None,
            });
        }
        RecordedBody::Sse { frames, done: true }
    } else {
        if !response_body.is_object() {
            return Err(external_error("non-streaming response body must be an object"));
        }
        RecordedBody::Json { value: response_body }
    };

    Ok(ImportedUpstream {
        source_id: Some(test_id),
        provider: None,
        model: Some(model),
        exchange: RecordedExchange {
            request: RecordedRequest {
                method,
                // `endpoint` is canonical because external recordings can have
                // a malformed duplicate path in `url`.
                path: endpoint,
                headers: safe_headers(headers),
                body: RecordedBody::Json { value: request_body },
            },
            response: RecordedResponse {
                status: response.status,
                headers: BTreeMap::from([("content-type".to_owned(), vec![response_content_type.to_owned()])]),
                body: response_body,
            },
        },
    })
}

fn unwrap_typed_wrappers(value: Value) -> Result<Value, FixtureError> {
    unwrap_typed_wrappers_at_depth(value, 0)
}

#[expect(
    clippy::too_many_lines,
    reason = "recursive wrapper validation keeps its complete shape check in one place"
)]
fn unwrap_typed_wrappers_at_depth(value: Value, depth: usize) -> Result<Value, FixtureError> {
    if depth > MAX_EXTERNAL_RESPONSE_DEPTH && matches!(&value, Value::Array(_) | Value::Object(_)) {
        return Err(external_error("external response nesting exceeds the depth limit"));
    }
    let child_depth = depth.saturating_add(1);
    match value {
        Value::Array(values) => values
            .into_iter()
            .map(|value| unwrap_typed_wrappers_at_depth(value, child_depth))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(mut values) => {
            let has_type = values.contains_key("__type__");
            let has_data = values.contains_key("__data__");
            if has_type || has_data {
                if !(has_type && has_data && values.len() == 2) {
                    return Err(external_error(
                        "typed wrapper must contain exactly __type__ and __data__",
                    ));
                }
                if !values.get("__type__").is_some_and(Value::is_string) {
                    return Err(external_error("typed wrapper __type__ must be a string"));
                }
                let data = values
                    .remove("__data__")
                    .ok_or_else(|| external_error("typed wrapper is missing __data__"))?;
                unwrap_typed_wrappers_at_depth(data, child_depth)
            } else {
                values
                    .into_iter()
                    .map(|(key, value)| unwrap_typed_wrappers_at_depth(value, child_depth).map(|value| (key, value)))
                    .collect::<Result<_, _>>()
                    .map(Value::Object)
            }
        },
        value => Ok(value),
    }
}

fn safe_headers(headers: BTreeMap<String, ExternalHeaderValues>) -> BTreeMap<String, Vec<String>> {
    let mut headers = headers
        .into_iter()
        .filter(|(name, _)| !is_credential_header(name))
        .map(|(name, values)| {
            let values = match values {
                ExternalHeaderValues::One(value) => vec![value],
                ExternalHeaderValues::Many(values) => values,
            };
            (name, values)
        })
        .collect::<BTreeMap<_, _>>();
    if !headers.keys().any(|name| name.eq_ignore_ascii_case("content-type")) {
        headers.insert("content-type".to_owned(), vec!["application/json".to_owned()]);
    }
    headers
}

/// Rejects response scripts that cannot represent a final body-bearing HTTP response.
fn validate_response_status(status: u16) -> Result<(), FixtureError> {
    if !(200..=599).contains(&status) {
        return Err(external_error("response status is not a supported final HTTP status"));
    }
    if matches!(status, 204 | 205 | 304) {
        return Err(external_error(
            "response status does not allow the required response body",
        ));
    }
    Ok(())
}


fn external_error(message: &'static str) -> FixtureError {
    FixtureError::ExternalRecording { message }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::{RecordedBody, import_external_recording};
    use crate::{Recording, ReplayTurn};

    const SECRET: &str = "very-secret-token";

    fn external_recording(response: &Value, is_streaming: bool) -> String {
        json!({
            "test_id": "source-test-id",
            "request": {
                "method": "POST",
                "endpoint": "/v1/chat/completions?stream=true",
                "url": "https://api.example.test/v1/v1/chat/completions?stream=true",
                "headers": {
                    "content-type": "application/json",
                    "x-trace": ["first", "second"],
                    "authorization": format!("Bearer {SECRET}")
                },
                "model": "example-model",
                "body": {"model": "example-model", "messages": [{"role": "user", "content": "hello"}]}
            },
            "response": {"body": response, "is_streaming": is_streaming},
            "id_normalization_mapping": {"chatcmpl_source": "chatcmpl_1"}
        })
        .to_string()
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the inline envelope makes the externally observed contract auditable"
    )]
    fn imports_non_streaming_wrapped_chat_completion() {
        // Catches a regression that drops request metadata or leaves provider wrappers in JSON bodies.
        let response = json!({
            "__type__": "openai.types.chat.chat_completion.ChatCompletion",
            "__data__": {
                "id": "chatcmpl_source",
                "object": "chat.completion",
                "choices": [{
                    "__type__": "provider.Choice",
                    "__data__": {"index": 0, "message": {"role": "assistant", "content": "hi"}}
                }]
            }
        });
        let content = external_recording(&response, false);

        let imported = import_external_recording(&content).expect("valid external recording should import");

        assert_eq!(imported.source_id.as_deref(), Some("source-test-id"));
        assert_eq!(imported.model.as_deref(), Some("example-model"));
        assert_eq!(imported.exchange.request.method, "POST");
        assert_eq!(imported.exchange.request.path, "/v1/chat/completions?stream=true");
        assert_eq!(
            imported.exchange.request.headers.get("content-type"),
            Some(&vec!["application/json".to_owned()])
        );
        assert_eq!(
            imported.exchange.request.headers.get("x-trace"),
            Some(&vec!["first".to_owned(), "second".to_owned()])
        );
        assert!(!imported.exchange.request.headers.contains_key("authorization"));
        assert_eq!(imported.exchange.response.status, 200);
        assert_eq!(
            imported.exchange.response.headers.get("content-type"),
            Some(&vec!["application/json".to_owned()])
        );
        let RecordedBody::Json { value } = &imported.exchange.request.body else {
            panic!("request body should be JSON");
        };
        assert_eq!(value["messages"][0]["content"], "hello");
        let RecordedBody::Json { value } = &imported.exchange.response.body else {
            panic!("non-streaming response should be JSON");
        };
        assert_eq!(value["choices"][0]["message"]["content"], "hi");
        assert!(value["choices"][0].get("__type__").is_none());
    }

    #[test]
    fn imports_explicit_response_status_and_defaults_omitted_status_to_ok() {
        // Catches discarding a truthful provider status while preserving old envelopes that omit it.
        let response = json!({"error": {"message": "slow down", "type": "rate_limit_error"}});
        let legacy = external_recording(&response, false);
        let mut explicit: Value = serde_json::from_str(&legacy).expect("inline fixture should be JSON");
        explicit["response"]["status"] = json!(429);

        let legacy = import_external_recording(&legacy).expect("legacy status-less envelope should import");
        let explicit = import_external_recording(&explicit.to_string()).expect("strict envelope status should import");

        assert_eq!(legacy.exchange.response.status, 200);
        assert_eq!(explicit.exchange.response.status, 429);
        assert_eq!(explicit.exchange.response.body, RecordedBody::Json { value: response });
    }

    #[test]
    fn response_status_accepts_only_supported_terminal_body_bearing_values() {
        // Catches deferring invalid response scripts to the later HTTP server boundary.
        let response = json!({"id": "response"});
        let base = external_recording(&response, false);

        for status in [0_u16, 99, 100, 199, 204, 205, 304, 600, u16::MAX] {
            let mut envelope: Value = serde_json::from_str(&base).expect("inline fixture should be JSON");
            envelope["response"]["status"] = json!(status);

            let error = import_external_recording(&envelope.to_string())
                .expect_err("unsupported terminal response status must fail during import");

            assert!(matches!(error, super::FixtureError::ExternalRecording { .. }));
            assert!(!error.to_string().contains(&status.to_string()));
        }

        for status in [200_u16, 201, 299, 300, 429, 599] {
            let mut envelope: Value = serde_json::from_str(&base).expect("inline fixture should be JSON");
            envelope["response"]["status"] = json!(status);

            let imported = import_external_recording(&envelope.to_string())
                .expect("supported terminal response status should import");

            assert_eq!(imported.exchange.response.status, status);
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the inline envelope keeps wrapper, header, frame-order, and DONE assertions auditable"
    )]
    fn imports_streaming_wrappers_as_ordered_done_sse_frames() {
        // Catches a regression that reorders stream chunks or omits the terminal completion marker.
        let response = json!([
            {"__type__": "openai.types.chat.chat_completion_chunk.ChatCompletionChunk", "__data__": {"id": "chunk_1", "choices": [{"delta": {"content": "hel"}}]}},
            {"__type__": "openai.types.chat.chat_completion_chunk.ChatCompletionChunk", "__data__": {"id": "chunk_1", "choices": [{"delta": {"content": "lo"}}]}}
        ]);
        let content = external_recording(&response, true);

        let imported = import_external_recording(&content).expect("valid stream should import");

        let body = &imported.exchange.response.body;
        assert_eq!(
            imported.exchange.response.headers.get("content-type"),
            Some(&vec!["text/event-stream".to_owned()])
        );
        assert!(
            body.render()
                .expect("canonical SSE should render")
                .ends_with(b"data: [DONE]\n\n")
        );
        let RecordedBody::Sse { frames, done } = body else {
            panic!("streaming response should be SSE");
        };
        assert!(done);
        assert_eq!(frames.len(), 2);
        assert_eq!(
            serde_json::from_str::<Value>(&frames[0].data).expect("first frame must be JSON")["choices"][0]["delta"]["content"],
            "hel"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&frames[1].data).expect("second frame must be JSON")["choices"][0]["delta"]["content"],
            "lo"
        );
        assert!(
            frames
                .iter()
                .all(|frame| frame.event.is_none() && frame.id.is_none() && frame.retry.is_none())
        );
    }

    #[test]
    fn rejects_malformed_or_extra_wrappers_without_leaking_body_or_secrets() {
        // Catches accepting ambiguous wrappers, which could silently change the recorded response shape.
        let response = json!({"__type__": "provider.Response", "__data__": {"token": SECRET}, "extra": true});
        let malformed = external_recording(&response, false);

        let error = import_external_recording(&malformed).expect_err("extra wrapper keys must be rejected");

        assert!(!error.to_string().contains(SECRET));
    }

    #[test]
    fn rejects_absent_request_body_and_response_streaming_mismatches() {
        // Catches producing an invalid canonical body when the external envelope contradicts itself.
        let response = json!({"ok": true});
        let mut absent_body: Value =
            serde_json::from_str(&external_recording(&response, false)).expect("inline fixture should be JSON");
        absent_body["request"]
            .as_object_mut()
            .expect("request should be an object")
            .remove("body");
        let absent_body = absent_body.to_string();
        assert!(import_external_recording(&absent_body).is_err());

        let response = json!([{ "__type__": "provider.Chunk", "__data__": {"ok": true} }]);
        let non_stream_array = external_recording(&response, false);
        assert!(import_external_recording(&non_stream_array).is_err());

        let response = json!({"__type__": "provider.Response", "__data__": {"ok": true}});
        let stream_object = external_recording(&response, true);
        assert!(import_external_recording(&stream_object).is_err());
    }

    #[test]
    fn non_streaming_responses_require_json_objects() {
        // Catches accepting a scalar or array as a provider response resource.
        let invalid = [
            ("null", Value::Null),
            ("string", Value::String(SECRET.to_owned())),
            ("number", json!(17)),
            ("boolean", json!(true)),
            ("array", json!([{"id": "response"}])),
            ("wrapped null", json!({"__type__": "label", "__data__": null})),
        ];

        for (case, response) in invalid {
            let content = external_recording(&response, false);
            let Err(error) = import_external_recording(&content) else {
                panic!("{case} non-streaming response should be rejected");
            };
            assert_eq!(
                error.to_string(),
                "invalid external recording: non-streaming response body must be an object"
            );
            assert!(!error.to_string().contains(SECRET));
        }
    }

    #[test]
    fn streaming_responses_require_nonempty_object_chunks() {
        // Catches manufacturing DONE-only streams or serializing scalar chunks as SSE frames.
        let invalid = [
            ("empty", json!([])),
            ("null chunk", json!([null])),
            ("string chunk", json!([SECRET])),
            ("number chunk", json!([17])),
            ("boolean chunk", json!([true])),
            ("nested array chunk", json!([[{"id": "chunk"}]])),
            (
                "wrapped scalar chunk",
                json!([{"__type__": "label", "__data__": SECRET}]),
            ),
            ("later invalid chunk", json!([{"id": "chunk"}, []])),
        ];

        for (case, response) in invalid {
            let content = external_recording(&response, true);
            let Err(error) = import_external_recording(&content) else {
                panic!("{case} streaming response should be rejected");
            };
            assert!(
                matches!(error, super::FixtureError::ExternalRecording { .. }),
                "{case} returned the wrong error: {error}"
            );
            assert!(!error.to_string().contains(SECRET));
        }
    }

    #[test]
    fn rejects_every_malformed_reserved_wrapper_shape_at_every_depth() {
        // Catches allowing a reserved wrapper key to hide inside an otherwise valid response object.
        let malformed = [
            ("type only", json!({"__type__": "label"})),
            ("data only", json!({"__data__": {"id": "response"}})),
            (
                "non-string type",
                json!({"__type__": 3, "__data__": {"id": "response"}}),
            ),
            (
                "extra key",
                json!({"__type__": "label", "__data__": {"id": "response"}, "extra": true}),
            ),
        ];

        for (shape, malformed) in malformed {
            let contexts = [
                ("top level", malformed.clone()),
                ("nested object", json!({"id": "response", "nested": malformed.clone()})),
                ("nested array", json!({"id": "response", "items": [malformed]})),
            ];
            for (context, response) in contexts {
                let content = external_recording(&response, false);
                let error = import_external_recording(&content)
                    .expect_err("malformed reserved wrapper shape should be rejected");
                assert!(
                    matches!(error, super::FixtureError::ExternalRecording { .. }),
                    "{shape} at {context} returned the wrong error: {error}"
                );
                assert!(!error.to_string().contains("response"));
            }
        }
    }

    #[test]
    fn exact_wrappers_unwrap_label_agnostically_at_every_depth() {
        // Catches adding a Python class-name dependency or stopping recursion at nested arrays.
        let response = json!({
            "__type__": "arbitrary.top.Label",
            "__data__": {
                "id": "response",
                "nested": {"__type__": "not.python.Nested", "__data__": {"value": 1}},
                "items": [{"__type__": "anything", "__data__": {"value": 2}}]
            }
        });

        let imported = import_external_recording(&external_recording(&response, false))
            .expect("exact wrappers with arbitrary labels should import");

        assert_eq!(
            imported.exchange.response.body,
            RecordedBody::Json {
                value: json!({"id": "response", "nested": {"value": 1}, "items": [{"value": 2}]})
            }
        );
    }

    #[test]
    fn typed_wrapper_unwrapping_rejects_values_beyond_the_depth_limit() {
        // Catches unbounded recursion when an external recording contains a deeply nested wrapper chain.
        let nested_wrappers = |depth| {
            (0..depth).fold(
                json!({"id": "response"}),
                |data, _| json!({"__type__": "provider.Wrapper", "__data__": data}),
            )
        };

        let accepted = super::unwrap_typed_wrappers(nested_wrappers(128))
            .expect("a wrapper chain at the documented depth ceiling should be accepted");
        let error = super::unwrap_typed_wrappers(nested_wrappers(129))
            .expect_err("a wrapper chain beyond the depth ceiling must be rejected");

        assert_eq!(accepted, json!({"id": "response"}));
        assert_eq!(
            error.to_string(),
            "invalid external recording: external response nesting exceeds the depth limit"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the table keeps every header preservation case visible at the importer boundary"
    )]
    fn filters_credentials_and_infers_content_type_without_overwriting_existing_values() {
        // Catches case-sensitive filtering, missing JSON inference, and content-type flattening or duplication.
        let response = json!({"id": "response"});
        let cases = [
            ("empty", json!({}), json!({"content-type": ["application/json"]})),
            (
                "safe header without content type",
                json!({"X-Trace": ["first", "second"]}),
                json!({
                    "X-Trace": ["first", "second"],
                    "content-type": ["application/json"]
                }),
            ),
            (
                "mixed-case existing content type",
                json!({"Content-Type": "application/problem+json", "X-Trace": "safe"}),
                json!({"Content-Type": ["application/problem+json"], "X-Trace": ["safe"]}),
            ),
            (
                "repeated existing content type",
                json!({"CONTENT-TYPE": ["application/json", "application/problem+json"]}),
                json!({"CONTENT-TYPE": ["application/json", "application/problem+json"]}),
            ),
            (
                "credential only",
                json!({
                    "AUTHORIZATION": format!("Bearer {SECRET}"),
                    "X-aPi-KeY": SECRET
                }),
                json!({"content-type": ["application/json"]}),
            ),
        ];

        for (case, headers, expected) in cases {
            let mut content: Value =
                serde_json::from_str(&external_recording(&response, false)).expect("inline fixture should be JSON");
            content["request"]["headers"] = headers;

            let imported = import_external_recording(&content.to_string()).expect("request headers should import");
            let expected = serde_json::from_value(expected).expect("expected headers should deserialize");

            assert_eq!(imported.exchange.request.headers, expected, "{case}");
        }
    }

    #[test]
    fn legacy_adapters_preserve_legacy_shapes_at_documented_lossy_boundaries() {
        // Catches changing legacy fixture serde while introducing the shared exchange representation.
        let recording: Recording = serde_json::from_value(json!({
            "source": "legacy recording",
            "request": {"model": "legacy-model"},
            "response": {"id": "legacy-response"}
        }))
        .expect("legacy recording should retain its on-disk shape");
        let recording_exchange = recording.to_recorded_exchange().expect("legacy recording should adapt");
        assert_eq!(recording_exchange.request.method, "POST");
        assert_eq!(recording_exchange.request.path, "");
        assert_eq!(recording_exchange.response.status, 200);
        assert!(matches!(recording_exchange.response.body, RecordedBody::Json { .. }));

        let turn: ReplayTurn = serde_json::from_value(json!({
            "name": "legacy-turn",
            "path": "/v1/responses",
            "request": {"model": "legacy-model"},
            "response": {"id": "response_legacy"}
        }))
        .expect("legacy replay turn should retain its on-disk shape");
        let turn_exchange = turn.to_client_exchange();
        assert_eq!(turn_exchange.request.method, "POST");
        assert_eq!(turn_exchange.request.path, "/v1/responses");
        assert_eq!(turn_exchange.response.status, 200);
        assert!(matches!(turn_exchange.request.body, RecordedBody::Json { .. }));
    }
}
