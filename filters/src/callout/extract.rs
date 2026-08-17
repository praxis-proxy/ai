// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `JSONPath` extraction and body shaping for the HTTP callout filter.

use std::collections::HashMap;

use praxis_filter::{FilterError, FilterResultSet};
use serde_json::Value;
use serde_json_path::JsonPath;
use tracing::{debug, warn};

// -----------------------------------------------------------------------------
// Compiled Extraction
// -----------------------------------------------------------------------------

/// A pre-compiled `JSONPath` extraction rule.
#[derive(Debug)]
pub(crate) struct CompiledExtraction {
    /// The compiled `JSONPath` expression.
    path: JsonPath,

    /// Key to write into [`FilterResultSet`].
    result_key: String,
}

impl CompiledExtraction {
    /// Parse and compile a `JSONPath` expression at config time.
    ///
    /// Also validates `result_key` up front by probing the result-set
    /// key rules, so invalid keys (empty, over-long, or containing
    /// characters other than ASCII alphanumerics, `_`, or `-`) fail at
    /// startup instead of silently on every request.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the expression is invalid or the
    /// result key is rejected by [`FilterResultSet`].
    pub(crate) fn compile(json_path: &str, result_key: String) -> Result<Self, FilterError> {
        let path = JsonPath::parse(json_path)
            .map_err(|e| -> FilterError { format!("http_callout: invalid JSONPath '{json_path}': {e}").into() })?;

        // Probe the key against the result-set rules with an empty value.
        FilterResultSet::new()
            .set(result_key.clone(), "")
            .map_err(|e| -> FilterError { format!("http_callout: invalid result_key '{result_key}': {e}").into() })?;

        Ok(Self { path, result_key })
    }

    /// Evaluate this extraction against a JSON value and write
    /// results into the result set.
    ///
    /// Coercion rules for the first matched node:
    /// - `bool` → `"true"` / `"false"`
    /// - `number` → decimal string
    /// - `string` → as-is
    /// - `array` / `object` → compact JSON
    /// - `null` or no match → skip (no entry written)
    ///
    /// The key is validated at config time (see [`Self::compile`]), so
    /// the only remaining failure mode is a coerced *value* rejected by
    /// the result-set limits (e.g. an oversized array/object coercion).
    /// Such a value is logged and skipped rather than failing the
    /// request, so a hostile or oversized third-party response is
    /// handled per the filter's `on_failure` policy instead of the `?`
    /// path.
    pub(crate) fn evaluate(&self, json: &Value, results: &mut FilterResultSet) {
        let node_list = self.path.query(json);
        let nodes: Vec<&Value> = node_list.all();

        let Some(first) = nodes.first() else {
            debug!(key = %self.result_key, "JSONPath matched no nodes; skipping");
            return;
        };

        let Some(value) = coerce_value(first) else {
            debug!(key = %self.result_key, "JSONPath matched null; skipping");
            return;
        };

        if let Err(error) = results.set(self.result_key.clone(), value) {
            warn!(
                key = %self.result_key,
                %error,
                "extracted value rejected by result set limits; skipping"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Body Shaping
// -----------------------------------------------------------------------------

/// Pre-compiled field→`JSONPath` mappings for reshaping the callout
/// request body.
///
/// When present, the downstream body is parsed as JSON and a new
/// object is constructed with only the mapped fields. The original
/// downstream body continues to the upstream untouched.
#[derive(Debug)]
pub(crate) struct BodyShaper {
    /// Compiled field mappings: `(output_field_name, jsonpath)`.
    fields: Vec<(String, JsonPath)>,
}

impl BodyShaper {
    /// Compile a set of field→`JSONPath` mappings at config time.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if any `JSONPath` expression is invalid.
    pub(crate) fn compile(mappings: &HashMap<String, String>) -> Result<Self, FilterError> {
        let mut fields = Vec::with_capacity(mappings.len());
        for (field, expr) in mappings {
            let path = JsonPath::parse(expr).map_err(|e| -> FilterError {
                format!("http_callout: invalid body JSONPath for field '{field}': {e}").into()
            })?;
            fields.push((field.clone(), path));
        }
        // Sort for deterministic output.
        fields.sort_by(|(a, _), (b, _)| a.cmp(b));
        Ok(Self { fields })
    }

    /// Whether any field mappings are configured.
    pub(crate) fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Reshape a raw body using the compiled `JSONPath` mappings.
    ///
    /// Parses `raw` as JSON, evaluates each mapping, and builds a
    /// new JSON object. Returns `None` if `raw` is not valid JSON.
    pub(crate) fn shape(&self, raw: &[u8]) -> Option<Vec<u8>> {
        let source: Value = serde_json::from_slice(raw).ok()?;
        let mut output = serde_json::Map::with_capacity(self.fields.len());

        for (field, path) in &self.fields {
            let node_list = path.query(&source);
            let nodes: Vec<&Value> = node_list.all();
            if let Some(value) = nodes.first() {
                output.insert(field.clone(), (*value).clone());
            }
        }

        serde_json::to_vec(&Value::Object(output)).ok()
    }
}

// -----------------------------------------------------------------------------
// Coercion
// -----------------------------------------------------------------------------

/// Coerce a JSON value to a string for [`FilterResultSet`].
///
/// Returns `None` for null values.
fn coerce_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => sanitize_string(s),
        Value::Array(_) | Value::Object(_) => sanitize_string(&value.to_string()),
    }
}

/// Maximum byte length for a sanitized value before truncation.
const MAX_SANITIZED_LEN: usize = 255;

/// Sanitize an extracted string value before it enters the result set.
///
/// Extracted values come from an untrusted third-party callout response
/// and feed [`FilterResultSet`] entries, branch matching, and logs, so
/// this is a defense-in-depth pass against control-character/log
/// injection. It:
///
/// - strips leading control characters and whitespace (e.g. a leading `"\n\n"`),
/// - keeps characters up to the first embedded control character and drops everything after it (multi-line verdicts
///   such as `"unsafe\nS02"` collapse to their first line, `"unsafe"`),
/// - drops `/` and `\` from the retained text,
/// - truncates the result to [`MAX_SANITIZED_LEN`] bytes at a UTF-8 character boundary, and
/// - returns `None` if nothing usable remains.
pub(crate) fn sanitize_string(raw: &str) -> Option<String> {
    // Skip leading control characters and whitespace (e.g. leading "\n\n").
    let trimmed = raw.trim_start_matches(|c: char| c < '\x20' || c == '\x7F' || c.is_whitespace());

    let (sanitized, rest) = split_at_first_control(trimmed);

    // Anything after the first embedded control character is dropped;
    // note it so a truncated multi-line value is not silently lost.
    let remaining = rest.trim();
    if !remaining.is_empty() {
        warn!(
            dropped = %remaining,
            kept = %sanitized,
            "extracted value contained control characters; truncated to first line"
        );
    }

    if sanitized.is_empty() {
        return None;
    }
    truncate_at_char_boundary(&sanitized, MAX_SANITIZED_LEN)
}

/// Split `input` at its first control character.
///
/// Returns `(kept, rest)` where `kept` is the text before the first
/// control character with `/` and `\` removed, and `rest` is the
/// remaining slice starting at that control character (empty if none).
fn split_at_first_control(input: &str) -> (String, &str) {
    let mut kept = String::with_capacity(input.len());
    for (idx, c) in input.char_indices() {
        if c < '\x20' || c == '\x7F' {
            // `char_indices` yields a valid boundary; `get` avoids the
            // deny-by-default `indexing_slicing` lint regardless.
            return (kept, input.get(idx..).unwrap_or(""));
        }
        if c != '/' && c != '\\' {
            kept.push(c);
        }
    }
    (kept, "")
}

/// Truncate `s` to at most `max` bytes on a UTF-8 boundary.
///
/// Returns `None` if the truncated result would be empty.
fn truncate_at_char_boundary(s: &str, max: usize) -> Option<String> {
    if s.len() <= max {
        return Some(s.to_owned());
    }
    let mut len = max;
    while len > 0 && !s.is_char_boundary(len) {
        len -= 1;
    }
    s.get(..len).filter(|t| !t.is_empty()).map(ToOwned::to_owned)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn compile_valid_expression() {
        assert!(
            CompiledExtraction::compile("$.flagged", "flagged".into()).is_ok(),
            "valid JSONPath should compile"
        );
    }

    #[test]
    fn compile_invalid_expression() {
        let err = CompiledExtraction::compile("$[invalid", "key".into()).unwrap_err();
        assert!(
            err.to_string().contains("invalid JSONPath"),
            "should report invalid expression: {err}"
        );
    }

    #[test]
    fn compile_rejects_invalid_result_key() {
        // A dotted key is rejected by the result-set key rules
        // (ASCII alphanumeric, '_', or '-' only) at config time.
        let err = CompiledExtraction::compile("$.flagged", "lakera.flagged".into()).unwrap_err();
        assert!(
            err.to_string().contains("invalid result_key"),
            "should report invalid result key: {err}"
        );
    }

    #[test]
    fn compile_rejects_empty_result_key() {
        let err = CompiledExtraction::compile("$.flagged", String::new()).unwrap_err();
        assert!(
            err.to_string().contains("invalid result_key"),
            "should report invalid result key: {err}"
        );
    }

    #[test]
    fn evaluate_bool_true() {
        let ext = CompiledExtraction::compile("$.flagged", "flagged".into()).unwrap();
        let json = json!({"flagged": true});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert_eq!(rs.get("flagged"), Some("true"));
    }

    #[test]
    fn evaluate_bool_false() {
        let ext = CompiledExtraction::compile("$.flagged", "flagged".into()).unwrap();
        let json = json!({"flagged": false});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert_eq!(rs.get("flagged"), Some("false"));
    }

    #[test]
    fn evaluate_number() {
        let ext = CompiledExtraction::compile("$.score", "score".into()).unwrap();
        let json = json!({"score": 0.95});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert_eq!(rs.get("score"), Some("0.95"));
    }

    #[test]
    fn evaluate_string() {
        let ext = CompiledExtraction::compile("$.label", "label".into()).unwrap();
        let json = json!({"label": "safe"});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert_eq!(rs.get("label"), Some("safe"));
    }

    #[test]
    fn evaluate_array() {
        let ext = CompiledExtraction::compile("$.tags", "tags".into()).unwrap();
        let json = json!({"tags": ["a", "b"]});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert_eq!(rs.get("tags"), Some(r#"["a","b"]"#));
    }

    #[test]
    fn evaluate_object() {
        let ext = CompiledExtraction::compile("$.meta", "meta".into()).unwrap();
        let json = json!({"meta": {"k": "v"}});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert_eq!(rs.get("meta"), Some(r#"{"k":"v"}"#));
    }

    #[test]
    fn evaluate_null_skips() {
        let ext = CompiledExtraction::compile("$.missing", "missing".into()).unwrap();
        let json = json!({"missing": null});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert!(rs.get("missing").is_none(), "null should be skipped");
    }

    #[test]
    fn evaluate_no_match_skips() {
        let ext = CompiledExtraction::compile("$.nonexistent", "key".into()).unwrap();
        let json = json!({"other": 1});
        let mut rs = FilterResultSet::new();
        ext.evaluate(&json, &mut rs);
        assert!(rs.get("key").is_none(), "no-match should be skipped");
    }

    // -------------------------------------------------------------------------
    // Sanitize String
    // -------------------------------------------------------------------------
    #[test]
    fn test_sanitize_string_leading_and_trailing_newlines() {
        // Leading newlines stripped, trailing category stripped -> "unsafe"
        let input = "\n\nunsafe\nS02";
        let result = sanitize_string(input);
        assert_eq!(result, Some("unsafe".to_owned()));
    }

    #[test]
    fn test_sanitize_string_safe() {
        let input = "safe";
        let result = sanitize_string(input);
        assert_eq!(result, Some("safe".to_owned()));
    }

    #[test]
    fn test_sanitize_string_empty_or_only_control_chars() {
        let input = "\n\n\r\t";
        let result = sanitize_string(input);
        assert_eq!(result, None);
    }

    // -------------------------------------------------------------------------
    // BodyShaper
    // -------------------------------------------------------------------------

    #[test]
    fn body_shaper_empty_mappings() {
        let shaper = BodyShaper::compile(&HashMap::new()).unwrap();
        assert!(shaper.is_empty(), "empty mappings should be empty");
    }

    #[test]
    fn body_shaper_picks_single_field() {
        let mut mappings = HashMap::new();
        mappings.insert("messages".into(), "$.messages".into());
        let shaper = BodyShaper::compile(&mappings).unwrap();

        let input = serde_json::to_vec(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let output = shaper.shape(&input).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed.get("messages").is_some(), "messages should be present");
        assert!(parsed.get("model").is_none(), "model should be stripped");
    }

    #[test]
    fn body_shaper_picks_multiple_fields() {
        let mut mappings = HashMap::new();
        mappings.insert("messages".into(), "$.messages".into());
        mappings.insert("stream".into(), "$.stream".into());
        let shaper = BodyShaper::compile(&mappings).unwrap();

        let input = serde_json::to_vec(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "temperature": 0.7
        }))
        .unwrap();

        let output = shaper.shape(&input).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed.get("messages").is_some(), "messages should be present");
        assert!(parsed.get("stream").is_some(), "stream should be present");
        assert!(parsed.get("model").is_none(), "model should be stripped");
        assert!(parsed.get("temperature").is_none(), "temperature should be stripped");
    }

    #[test]
    fn body_shaper_missing_field_omitted() {
        let mut mappings = HashMap::new();
        mappings.insert("messages".into(), "$.messages".into());
        mappings.insert("absent".into(), "$.nonexistent".into());
        let shaper = BodyShaper::compile(&mappings).unwrap();

        let input = serde_json::to_vec(&json!({
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let output = shaper.shape(&input).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed.get("messages").is_some(), "messages should be present");
        assert!(parsed.get("absent").is_none(), "missing field should be omitted");
    }

    #[test]
    fn body_shaper_invalid_json_returns_none() {
        let mut mappings = HashMap::new();
        mappings.insert("x".into(), "$.x".into());
        let shaper = BodyShaper::compile(&mappings).unwrap();

        assert!(shaper.shape(b"not json").is_none(), "invalid JSON should return None");
    }

    #[test]
    fn body_shaper_invalid_jsonpath_rejected() {
        let mut mappings = HashMap::new();
        mappings.insert("x".into(), "$[invalid".into());
        let err = BodyShaper::compile(&mappings).expect_err("expected error");
        assert!(
            err.to_string().contains("invalid body JSONPath"),
            "should report invalid JSONPath: {err}"
        );
    }

    #[test]
    fn body_shaper_nested_extraction() {
        let mut mappings = HashMap::new();
        mappings.insert("content".into(), "$.messages[0].content".into());
        let shaper = BodyShaper::compile(&mappings).unwrap();

        let input = serde_json::to_vec(&json!({
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .unwrap();

        let output = shaper.shape(&input).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["content"], "hello world", "should extract nested value");
    }
}
