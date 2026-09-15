// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Azure OpenAI request transformation.
//!
//! Azure OpenAI accepts Chat Completions request bodies as-is. The only
//! body-level normalization is stripping the `model` field, which Azure
//! ignores (the deployment name in the URL determines the model). Removing
//! it avoids operator confusion when the body `model` and deployment
//! diverge.

use serde_json::Value;

/// Strip Azure-irrelevant fields from a Chat Completions request body.
///
/// Returns `Some(transformed_bytes)` when the body was modified, or
/// `None` when no changes were necessary (callers should keep the
/// original).
pub(crate) fn strip_ignored_fields(body: &[u8]) -> Option<Vec<u8>> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    let obj = value.as_object_mut()?;

    obj.remove("model")?;

    serde_json::to_vec(&value).ok()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn strips_model_field() {
        let input = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0.7}"#;
        let result = strip_ignored_fields(input).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(parsed.get("model").is_none(), "model should be removed");
        assert_eq!(parsed["messages"][0]["role"], "user", "other fields preserved");
        assert_eq!(parsed["temperature"], 0.7);
    }

    #[test]
    fn returns_none_without_model() {
        let input = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        assert!(strip_ignored_fields(input).is_none());
    }

    #[test]
    fn returns_none_for_invalid_json() {
        assert!(strip_ignored_fields(b"not json").is_none());
    }
}
