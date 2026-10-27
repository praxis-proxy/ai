// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Current-request serialization for Responses state (#1618).
//!
//! `from_request_body` moves `input` out of the parsed parameters so the payload
//! is decoded only once. [`InputForm`] records the input's original JSON shape so
//! callers can reproduce the client's form (including the string shorthand)
//! without retaining a second decoded copy: MCP tool discovery re-serializes the
//! current request via `CurrentRequestBody` (gated on `openai-mcp-tools`), and
//! response synthesis echoes the client's input back onto the Responses resource
//! via [`input_echo_value`].

#[cfg(feature = "openai-mcp-tools")]
use serde::ser::SerializeMap as _;
use serde_json::Value;

use super::state::ResponsesState;

/// Original JSON shape of a request's `input`, retained so current-request
/// serialization reproduces the client's form without a second owned copy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum InputForm {
    /// No `input` member was present.
    #[default]
    Missing,
    /// `input` was JSON `null`.
    Null,
    /// `input` was a string shorthand.
    String,
    /// `input` was a single message object.
    Object,
    /// `input` was an array of items.
    Array,
}

impl InputForm {
    /// Classify a request's `input` value, or its absence.
    ///
    /// Unsupported scalars (number/bool) map to [`Self::Missing`] because they are
    /// left in the parameter object rather than tracked here, so the current-request
    /// view re-emits them once from the parameters without duplication (#1618).
    #[must_use]
    pub(crate) fn classify(input: Option<&Value>) -> Self {
        match input {
            Some(Value::Null) => Self::Null,
            Some(Value::String(_)) => Self::String,
            Some(Value::Object(_)) => Self::Object,
            Some(Value::Array(_)) => Self::Array,
            None | Some(_) => Self::Missing,
        }
    }
}

/// Serializes the current request: parameters plus the current input in its
/// original form. Borrows state and never reconstructs an owned body.
///
/// Only MCP tool discovery re-serializes the current request, so this view is
/// gated on `openai-mcp-tools`; [`input_echo_value`] serves the always-present
/// response-synthesis echo.
#[cfg(feature = "openai-mcp-tools")]
pub(crate) struct CurrentRequestBody<'a> {
    /// Request state to project.
    state: &'a ResponsesState,
}

#[cfg(feature = "openai-mcp-tools")]
impl<'a> CurrentRequestBody<'a> {
    /// Borrow `state` for current-request serialization.
    #[must_use]
    pub(crate) fn new(state: &'a ResponsesState) -> Self {
        Self { state }
    }
}

#[cfg(feature = "openai-mcp-tools")]
impl serde::Serialize for CurrentRequestBody<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Some(object) = self.state.request_body.as_object() else {
            return self.state.request_body.serialize(serializer);
        };
        let mut map = serializer.serialize_map(None)?;
        for (name, value) in object {
            map.serialize_entry(name, value)?;
        }
        let input = &self.state.input;
        match self.state.input_form {
            InputForm::Missing => {},
            InputForm::Null => map.serialize_entry("input", &Value::Null)?,
            // The normalized string item keeps the original text under `content`.
            InputForm::String => {
                let content = input
                    .get(0)
                    .and_then(|item| item.get("content"))
                    .unwrap_or(&Value::Null);
                map.serialize_entry("input", content)?;
            },
            InputForm::Object => map.serialize_entry("input", input.get(0).unwrap_or(&Value::Null))?,
            InputForm::Array => map.serialize_entry("input", input)?,
        }
        map.end()
    }
}

/// Reconstruct the client's `input` in its original JSON shape for echoing onto
/// a synthesized Responses resource.
///
/// `from_request_body` strips `input` from `request_body` (#1618), so response
/// synthesis can no longer borrow it there. This rebuilds the echo from the
/// shared history and the recorded [`InputForm`] — returning [`None`] when the
/// request carried no `input` member, so the resource field stays `null` exactly
/// as it did when the value was read straight from `request_body`. The returned
/// value owns a single copy that the resource serializer then clones once per
/// snapshot, matching the pre-#1618 cost.
#[must_use]
pub(crate) fn input_echo_value(state: &ResponsesState) -> Option<Value> {
    let input = &state.input;
    match state.input_form {
        InputForm::Missing => None,
        InputForm::Null => Some(Value::Null),
        // The normalized string item keeps the original text under `content`.
        InputForm::String => Some(
            input
                .get(0)
                .and_then(|item| item.get("content"))
                .cloned()
                .unwrap_or(Value::Null),
        ),
        InputForm::Object => Some(input.get(0).cloned().unwrap_or(Value::Null)),
        InputForm::Array => Some(Value::Array(input.iter().cloned().collect())),
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    /// Serialize the current-request view of `body` and return its `input` member.
    #[cfg(feature = "openai-mcp-tools")]
    fn current_input(body: Value) -> Option<Value> {
        let state = ResponsesState::from_request_body(body);
        let serialized = serde_json::to_value(CurrentRequestBody::new(&state)).unwrap();
        serialized.get("input").cloned()
    }

    /// Reconstruct the response echo for `body` from its built state.
    fn echo_input(body: Value) -> Option<Value> {
        input_echo_value(&ResponsesState::from_request_body(body))
    }

    #[test]
    fn echo_rebuilds_string_shorthand() {
        assert_eq!(
            echo_input(json!({"model": "m", "input": "hi"})),
            Some(json!("hi")),
            "string shorthand echoes the original text, not the normalized item"
        );
    }

    #[test]
    fn echo_rebuilds_single_object() {
        let object = json!({"type": "message", "role": "user", "content": "hi"});
        assert_eq!(
            echo_input(json!({"model": "m", "input": object.clone()})),
            Some(object),
            "object input echoes as a single object"
        );
    }

    #[test]
    fn echo_rebuilds_array() {
        let array = json!([{"type": "message", "role": "user", "content": "a"}]);
        assert_eq!(
            echo_input(json!({"model": "m", "input": array.clone()})),
            Some(array),
            "array input echoes as an array"
        );
    }

    #[test]
    fn echo_rebuilds_null() {
        assert_eq!(
            echo_input(json!({"model": "m", "input": null})),
            Some(json!(null)),
            "an explicit null input echoes as null"
        );
    }

    #[test]
    fn echo_omits_missing_input() {
        assert_eq!(
            echo_input(json!({"model": "m"})),
            None,
            "a missing input leaves the resource field at its absent default"
        );
    }

    #[test]
    fn echo_omits_unsupported_scalar() {
        // A scalar is not tracked in `input_form`; it stays in `request_body`, so
        // the echo falls back to the absent default here and the resource reads it
        // straight from the parameters instead (#1618).
        assert_eq!(
            echo_input(json!({"model": "m", "input": 42})),
            None,
            "an unsupported scalar is not tracked, so the echo omits it"
        );
    }

    #[cfg(feature = "openai-mcp-tools")]
    #[test]
    fn string_input_round_trips_as_shorthand() {
        assert_eq!(
            current_input(json!({"model": "m", "input": "hi"})),
            Some(json!("hi")),
            "string shorthand must be preserved without a second copy"
        );
    }

    #[cfg(feature = "openai-mcp-tools")]
    #[test]
    fn object_input_round_trips_as_single_object() {
        let object = json!({"type": "message", "role": "user", "content": "hi"});
        assert_eq!(
            current_input(json!({"model": "m", "input": object.clone()})),
            Some(object),
            "object input must round-trip as a single object"
        );
    }

    #[cfg(feature = "openai-mcp-tools")]
    #[test]
    fn array_input_round_trips_as_array() {
        let array = json!([{"type": "message", "role": "user", "content": "a"}]);
        assert_eq!(
            current_input(json!({"model": "m", "input": array.clone()})),
            Some(array),
            "array input must round-trip as an array"
        );
    }

    #[cfg(feature = "openai-mcp-tools")]
    #[test]
    fn null_input_round_trips_as_null() {
        assert_eq!(current_input(json!({"model": "m", "input": null})), Some(json!(null)));
    }

    #[cfg(feature = "openai-mcp-tools")]
    #[test]
    fn missing_input_is_omitted() {
        assert_eq!(
            current_input(json!({"model": "m"})),
            None,
            "a missing input stays missing"
        );
    }

    #[cfg(feature = "openai-mcp-tools")]
    #[test]
    fn unsupported_scalar_input_stays_in_parameters() {
        // A number cannot be normalized; it is retained in the parameters and
        // emitted once so existing validators still reject it (#1618).
        assert_eq!(current_input(json!({"model": "m", "input": 42})), Some(json!(42)));
    }

    #[test]
    fn classify_maps_each_shape() {
        assert_eq!(InputForm::classify(None), InputForm::Missing);
        assert_eq!(InputForm::classify(Some(&json!(null))), InputForm::Null);
        assert_eq!(InputForm::classify(Some(&json!("s"))), InputForm::String);
        assert_eq!(InputForm::classify(Some(&json!({}))), InputForm::Object);
        assert_eq!(InputForm::classify(Some(&json!([]))), InputForm::Array);
        assert_eq!(InputForm::classify(Some(&json!(42))), InputForm::Missing);
    }
}
