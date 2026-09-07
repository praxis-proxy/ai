// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Load recording fixtures for integration tests.
//!
//! A recording captures a request/response pair from an AI API
//! interaction.  Non-streaming recordings use `response` (a JSON
//! object); streaming recordings use `response_sse` (raw SSE text).

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::inference_fixture::{FixtureError, RecordedBody, RecordedExchange, RecordedRequest, RecordedResponse};

/// A recorded API request/response pair loaded from a JSON fixture.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recording {
    /// Human-readable description of this recording.
    pub source: String,
    /// The request body sent to the API.
    pub request: serde_json::Value,
    /// Non-streaming JSON response body.
    #[serde(default)]
    pub response: Option<serde_json::Value>,
    /// Streaming SSE response body (raw text).
    #[serde(default)]
    pub response_sse: Option<String>,
}

impl Recording {
    /// Load a recording from a fixture file relative to
    /// `tests/integration/fixtures/`.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be read or parsed, or if neither
    /// `response` nor `response_sse` is present.
    pub fn load(relative_path: &str) -> Self {
        let base = format!("{}/../integration/fixtures/{relative_path}", env!("CARGO_MANIFEST_DIR"),);
        let content = std::fs::read_to_string(&base).unwrap_or_else(|e| panic!("read fixture {base}: {e}"));
        let recording: Self =
            serde_json::from_str(&content).unwrap_or_else(|e| panic!("parse fixture {relative_path}: {e}"));
        assert!(
            recording.response.is_some() || recording.response_sse.is_some(),
            "fixture {relative_path} must have `response` or `response_sse`"
        );
        recording
    }

    /// Return the request body as a compact JSON string.
    ///
    /// # Panics
    ///
    /// Panics if the request value cannot be serialized.
    pub fn request_body(&self) -> String {
        serde_json::to_string(&self.request).unwrap_or_else(|e| panic!("serialize request: {e}"))
    }

    /// Compact JSON for non-streaming, raw SSE text for streaming.
    ///
    /// # Panics
    ///
    /// Panics if neither `response` nor `response_sse` is set, or if
    /// the response value cannot be serialized.
    pub fn response_body(&self) -> String {
        if let Some(sse) = &self.response_sse {
            sse.clone()
        } else if let Some(resp) = &self.response {
            serde_json::to_string(resp).unwrap_or_else(|e| panic!("serialize response: {e}"))
        } else {
            panic!("recording has neither response nor response_sse")
        }
    }

    /// Whether this recording uses SSE streaming.
    pub fn is_streaming(&self) -> bool {
        self.response_sse.is_some()
    }

    /// Convert this legacy recording into the shared exchange schema.
    ///
    /// Legacy recording files do not include an HTTP method or path, so this
    /// lossy adapter always uses `POST` and an empty path. Callers must supply
    /// the route when replaying the converted exchange.
    ///
    /// The adapter borrows the legacy recording, so JSON values are cloned at
    /// the owned shared-schema boundary; raw SSE is parsed from the borrowed
    /// string without cloning it.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::LegacyRecordingMissingResponse`] if neither
    /// legacy response representation is present.
    ///
    /// Raw SSE delegates to the shared body parser. The legacy `String` field
    /// guarantees UTF-8, so its current SSE error path is unrepresentable; the
    /// `Result` preserves the shared parser boundary for future validation.
    pub fn to_recorded_exchange(&self) -> Result<RecordedExchange, FixtureError> {
        let response_body = if let Some(response_sse) = &self.response_sse {
            RecordedBody::from_http(Some("text/event-stream"), response_sse.as_bytes())?
        } else if let Some(response) = &self.response {
            RecordedBody::Json {
                // The converted exchange owns its body while `self` remains usable.
                value: response.clone(),
            }
        } else {
            return Err(FixtureError::LegacyRecordingMissingResponse);
        };

        Ok(RecordedExchange {
            request: RecordedRequest {
                method: "POST".to_owned(),
                path: String::new(),
                headers: BTreeMap::new(),
                // The converted exchange owns its body while `self` remains usable.
                body: RecordedBody::Json {
                    value: self.request.clone(),
                },
            },
            response: RecordedResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: response_body,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn load_non_streaming() {
        let r = Recording::load("anthropic/messages/basic.json");
        assert!(!r.is_streaming(), "basic fixture should be non-streaming");
        assert!(r.response.is_some(), "basic fixture should include a JSON response");
        assert!(
            r.response_sse.is_none(),
            "basic fixture should not include an SSE response"
        );
    }

    #[test]
    fn load_streaming() {
        let r = Recording::load("anthropic/messages/streaming_basic.json");
        assert!(r.is_streaming(), "streaming fixture should be streaming");
        assert!(
            r.response_sse.is_some(),
            "streaming fixture should include an SSE response"
        );
        assert!(
            r.response.is_none(),
            "streaming fixture should not include a JSON response"
        );
    }

    #[test]
    fn request_body_is_valid_json() {
        let r = Recording::load("anthropic/messages/basic.json");
        let body = r.request_body();
        serde_json::from_str::<serde_json::Value>(&body).expect("request_body should be valid JSON");
    }

    #[test]
    fn response_body_non_streaming_is_valid_json() {
        let r = Recording::load("anthropic/messages/basic.json");
        let body = r.response_body();
        serde_json::from_str::<serde_json::Value>(&body).expect("non-streaming response_body should be valid JSON");
    }

    #[test]
    fn response_body_streaming_contains_sse_events() {
        let r = Recording::load("anthropic/messages/streaming_basic.json");
        let body = r.response_body();
        assert!(body.contains("event:"), "streaming response should contain SSE events");
    }

    #[test]
    fn adapter_preserves_body_direction_and_lossy_http_defaults() {
        // Catches swapping request and response bodies at the compatibility boundary.
        let recording = Recording {
            source: "legacy source".to_owned(),
            request: json!({"request_model": "request-only-model"}),
            response: Some(json!({"response_id": "response-only-id", "content": "answer"})),
            response_sse: None,
        };

        let exchange = recording.to_recorded_exchange().expect("recording should adapt");

        assert_eq!(exchange.request.method, "POST");
        assert_eq!(exchange.request.path, "");
        assert!(exchange.request.headers.is_empty());
        assert_eq!(
            exchange.request.body,
            RecordedBody::Json {
                value: json!({"request_model": "request-only-model"})
            }
        );
        assert_eq!(exchange.response.status, 200);
        assert!(exchange.response.headers.is_empty());
        assert_eq!(
            exchange.response.body,
            RecordedBody::Json {
                value: json!({"response_id": "response-only-id", "content": "answer"})
            }
        );
    }

    #[test]
    fn adapter_parses_and_canonically_renders_raw_sse() {
        // Catches treating legacy raw SSE as JSON or dropping event metadata and DONE.
        let raw = "event: message_delta\r\ndata: {\"type\":\"message_delta\",\"delta\":{\"text\":\"hi\"}}\r\n\r\ndata: [DONE]\r\n\r\n";
        let recording = Recording {
            source: "legacy SSE".to_owned(),
            request: json!({"stream": true}),
            response: None,
            response_sse: Some(raw.to_owned()),
        };

        let exchange = recording.to_recorded_exchange().expect("valid SSE should adapt");

        let RecordedBody::Sse { frames, done } = &exchange.response.body else {
            panic!("legacy SSE should become canonical SSE");
        };
        assert_eq!(
            frames,
            &[crate::inference_fixture::SseFrame {
                event: Some("message_delta".to_owned()),
                data: r#"{"type":"message_delta","delta":{"text":"hi"}}"#.to_owned(),
                id: None,
                retry: None,
            }]
        );
        assert!(*done);
        assert_eq!(
            exchange.response.body.render().expect("SSE should render"),
            b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"text\":\"hi\"}}\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn adapter_reports_legacy_missing_response_without_leaking_source_or_request() {
        // Catches reusing an external-import error or reflecting legacy payloads in diagnostics.
        let recording = Recording {
            source: "secret-source-name".to_owned(),
            request: json!({"secret": "secret-request-payload"}),
            response: None,
            response_sse: None,
        };

        let error = recording
            .to_recorded_exchange()
            .expect_err("missing legacy response should fail");

        assert_eq!(error.to_string(), "legacy recording is missing a response body");
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn legacy_recording_json_shape_keeps_optional_response_defaults() {
        // Catches making either legacy response representation required or accepting new fields.
        let full_shape = json!({
            "source": "legacy source",
            "request": {"request_marker": "request"},
            "response": {"response_marker": "response"},
            "response_sse": null
        });
        let recording: Recording =
            serde_json::from_value(full_shape).expect("complete legacy recording shape should deserialize");
        assert_eq!(recording.source, "legacy source");
        assert_eq!(recording.request_body(), r#"{"request_marker":"request"}"#);
        assert_eq!(recording.response_body(), r#"{"response_marker":"response"}"#);
        assert!(recording.response_sse.is_none());

        let minimal_shape = json!({
            "source": "legacy source",
            "request": {"request_marker": "request"}
        });

        let recording: Recording =
            serde_json::from_value(minimal_shape).expect("legacy optional fields should default");

        assert_eq!(recording.source, "legacy source");
        assert_eq!(recording.request, json!({"request_marker": "request"}));
        assert_eq!(recording.response, None);
        assert_eq!(recording.response_sse, None);
        assert!(
            serde_json::from_value::<Recording>(json!({
                "source": "legacy source",
                "request": {},
                "unexpected": true
            }))
            .is_err()
        );
    }
}
