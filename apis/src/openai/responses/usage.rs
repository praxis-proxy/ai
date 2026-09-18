// SPDX-License-Identifier: Apache-2.0
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
#[cfg(test)]
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

/// Saturating recursive sum that consumes the current usage tree.
///
/// This is the ownership-preserving counterpart to the borrowed merge helper. Newly
/// introduced fields and replacement values are moved into the accumulator,
/// avoiding a second full usage owner while an agentic round is finalized.
pub(crate) fn merge_usage_owned(accumulated: &mut Value, current: Value) {
    match (accumulated, current) {
        (Value::Object(accumulated), Value::Object(current)) => {
            for (key, value) in current {
                match accumulated.get_mut(&key) {
                    Some(existing) => merge_usage_owned(existing, value),
                    None => {
                        accumulated.insert(key, value);
                    },
                }
            }
        },
        (Value::Number(accumulated), Value::Number(current)) => {
            if let (Some(left), Some(right)) = (accumulated.as_u64(), current.as_u64()) {
                *accumulated = serde_json::Number::from(left.saturating_add(right));
            } else {
                *accumulated = current;
            }
        },
        (accumulated, current) => *accumulated = current,
    }
}

/// Compact JSON bytes of the value produced by [`merge_usage_owned`] without
/// constructing a second usage tree.
#[expect(
    clippy::too_many_lines,
    reason = "mirrors recursive object union and saturating numeric merge"
)]
pub(crate) fn merged_usage_json_bytes(accumulated: &Value, current: &Value) -> Option<usize> {
    match (accumulated, current) {
        (Value::Object(accumulated), Value::Object(current)) => {
            let mut bytes = 2_usize;
            let mut entries = 0_usize;
            for (key, value) in accumulated {
                let value_bytes = current.get(key).map_or_else(
                    || super::state::retained_json_bytes(value),
                    |next| merged_usage_json_bytes(value, next),
                )?;
                bytes = bytes
                    .checked_add(super::state::retained_json_bytes(key)?)?
                    .checked_add(1)?
                    .checked_add(value_bytes)?;
                entries = entries.checked_add(1)?;
            }
            for (key, value) in current {
                if accumulated.contains_key(key) {
                    continue;
                }
                bytes = bytes
                    .checked_add(super::state::retained_json_bytes(key)?)?
                    .checked_add(1)?
                    .checked_add(super::state::retained_json_bytes(value)?)?;
                entries = entries.checked_add(1)?;
            }
            bytes.checked_add(entries.saturating_sub(1))
        },
        (Value::Number(left), Value::Number(right)) => {
            if let (Some(left), Some(right)) = (left.as_u64(), right.as_u64()) {
                Some(decimal_len(left.saturating_add(right)))
            } else {
                super::state::retained_json_bytes(current)
            }
        },
        _ => super::state::retained_json_bytes(current),
    }
}

/// Return the number of decimal digits in an unsigned integer.
fn decimal_len(value: u64) -> usize {
    if value == 0 {
        1
    } else {
        usize::try_from(value.ilog10()).unwrap_or(19) + 1
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::{merge_usage, merged_usage_json_bytes};

    #[test]
    fn projected_owned_merge_size_matches_compact_json() {
        let mut accumulated = json!({
            "input_tokens": 99,
            "input_tokens_details": {"cached_tokens": 4},
            "replaced": "old"
        });
        let current = json!({
            "input_tokens": 1,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 6, "audio_tokens": 1},
            "replaced": null
        });
        let projected = merged_usage_json_bytes(&accumulated, &current).unwrap();
        super::merge_usage_owned(&mut accumulated, current);
        assert_eq!(
            projected,
            crate::openai::responses::state::retained_json_bytes(&accumulated).unwrap()
        );
    }

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
