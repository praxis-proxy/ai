// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Usage accumulation for the Responses API.
//!
//! Centralises the recursive merge so every iterative flow
//! (streaming, agentic loop, file search) shares one implementation
//! rather than coupling non-streaming filters to the SSE-specific
//! `stream_events` module.

use serde_json::Value;

/// Saturating recursive sum for numeric token-usage fields.
///
/// Recursively merges `current` into `accumulated`:
/// - Objects: fields are merged recursively; newly introduced keys are preserved.
/// - Unsigned integers: saturating-add.
/// - All other types (including signed/float numbers and type mismatches): `accumulated` is replaced with `current`.
pub(crate) fn merge_usage(accumulated: &mut Value, current: &Value) {
    match (accumulated, current) {
        (Value::Object(accumulated), Value::Object(current)) => {
            for (key, value) in current {
                match accumulated.get_mut(key) {
                    Some(existing) => merge_usage(existing, value),
                    None => {
                        accumulated.insert(key.clone(), value.clone());
                    },
                }
            }
        },
        (Value::Number(accumulated), Value::Number(current)) => {
            if let (Some(left), Some(right)) = (accumulated.as_u64(), current.as_u64()) {
                *accumulated = serde_json::Number::from(left.saturating_add(right));
            } else {
                *accumulated = current.clone();
            }
        },
        (accumulated, current) => current.clone_into(accumulated),
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::merge_usage;

    #[test]
    fn adds_unsigned_integer_fields() {
        let mut acc = json!({"input_tokens": 10, "output_tokens": 5});
        merge_usage(&mut acc, &json!({"input_tokens": 7, "output_tokens": 1}));
        assert_eq!(acc["input_tokens"], 17);
        assert_eq!(acc["output_tokens"], 6);
    }

    #[test]
    fn preserves_new_keys() {
        let mut acc = json!({"input_tokens": 10});
        merge_usage(&mut acc, &json!({"output_tokens": 5}));
        assert_eq!(acc["input_tokens"], 10);
        assert_eq!(acc["output_tokens"], 5);
    }

    #[test]
    fn merges_nested_objects() {
        let mut acc = json!({"input_tokens_details": {"cached_tokens": 4}});
        merge_usage(
            &mut acc,
            &json!({"input_tokens_details": {"cached_tokens": 2, "audio_tokens": 1}}),
        );
        assert_eq!(acc["input_tokens_details"]["cached_tokens"], 6);
        assert_eq!(acc["input_tokens_details"]["audio_tokens"], 1);
    }

    #[test]
    fn saturates_at_u64_max() {
        let mut acc = json!({"input_tokens": u64::MAX});
        merge_usage(&mut acc, &json!({"input_tokens": 1}));
        assert_eq!(acc["input_tokens"], u64::MAX);
    }

    #[test]
    fn replaces_on_type_mismatch() {
        let mut acc = json!({"field": "old_string"});
        merge_usage(&mut acc, &json!({"field": 42}));
        assert_eq!(acc["field"], 42);
    }

    #[test]
    fn replaces_non_unsigned_number() {
        let mut acc = json!({"field": -1});
        merge_usage(&mut acc, &json!({"field": 5}));
        assert_eq!(
            acc["field"], 5,
            "signed/float numbers should replace rather than saturating-add"
        );
    }

    #[test]
    fn null_accumulated_replaced_by_object() {
        let mut acc = serde_json::Value::Null;
        merge_usage(&mut acc, &json!({"input_tokens": 3}));
        assert_eq!(acc["input_tokens"], 3);
    }
}
