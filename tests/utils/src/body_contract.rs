// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-body preservation checks, independent of a particular filter.
//!
//! A no-op must preserve bytes. A JSON mutation may normalize whitespace,
//! escapes, and number spelling, but must preserve values and object-member
//! order outside its declared changes. These are different contracts: parsing
//! both bodies and comparing `Value`s cannot establish byte preservation, and
//! `Value` equality alone ignores object-member order.

use serde_json::{Map, Value};

/// Assert that a no-op forwarded the exact original HTTP body content.
///
/// HTTP headers, chunk boundaries, and transfer framing are outside this check.
///
/// # Panics
///
/// Panics if the body differs. Diagnostics omit the body contents.
pub fn assert_body_unchanged(before: &[u8], after: &[u8]) {
    assert!(before == after, "request body changed on a no-op path");
}

/// Assert that only the listed JSON pointers may change.
///
/// Pointers name complete subtrees, including fields added or removed by the
/// filter. Values, array order, and the relative order of object members outside
/// those subtrees must remain unchanged. A separate assertion should verify the
/// intended new values; this check guards everything the filter does not own.
///
/// Whitespace, escapes, and numeric spelling are not compared. Use
/// [`assert_body_unchanged`] when the contract is exact body bytes.
///
/// # Panics
///
/// Panics for invalid JSON, invalid/root/unused change pointers, or an undeclared
/// change. Diagnostics omit payload values and data-controlled field names.
pub fn assert_json_changes_only(before: &[u8], after: &[u8], changes: &[&str]) {
    let before: Value = serde_json::from_slice(before).expect("original body must be JSON");
    let after: Value = serde_json::from_slice(after).expect("forwarded body must be JSON");
    for pointer in changes {
        assert!(
            pointer.starts_with('/') && valid_pointer(pointer),
            "a change must name a non-root JSON pointer"
        );
        assert!(
            before.pointer(pointer).is_some() || after.pointer(pointer).is_some(),
            "change pointer is absent from both bodies"
        );
    }
    compare_unchanged(&before, &after, changes, "", "body");
}

/// Find a change to the relative order of semantically retained object members.
///
/// This complements a transformation's expected-output assertions. It compares
/// corresponding fields and array positions even when sibling values change.
/// Added and removed keys are ignored, so moved/translated content
/// still needs a filter-specific mapping test. Call before fixture normalization,
/// which intentionally sorts keys and would erase this evidence.
///
/// Returns an opaque position rather than payload values or field names.
#[must_use]
pub fn retained_json_order_mismatch(before: &Value, after: &Value) -> Option<String> {
    retained_order(before, after, "body", true)
}

/// Endpoint translations own their protocol envelopes; intact subtrees still retain order.
pub(crate) fn translated_json_order_mismatch(before: &Value, after: &Value) -> Option<String> {
    retained_order(before, after, "body", false)
}

/// Check retained function-tool order and schema property order across the
/// OpenAI, Anthropic, Bedrock, and Gemini tool envelopes.
///
/// Matching uses tool names, so changing a provider wrapper does not hide the
/// original schema. Provider-specific value/type conversions remain governed
/// by the transformation's expected-output tests. Added/removed server tools
/// are allowed; retained function tools must keep their relative order.
#[must_use]
pub fn retained_tool_order_mismatch(before: &Value, after: &Value) -> Option<String> {
    let before = tool_schemas(before);
    let after = tool_schemas(after);
    let before_names = before
        .iter()
        .map(|(name, _)| name)
        .filter(|name| after.iter().any(|(other, _)| other == *name));
    let after_names = after
        .iter()
        .map(|(name, _)| name)
        .filter(|name| before.iter().any(|(other, _)| other == *name));
    if !before_names.eq(after_names) {
        return Some("body.tools".to_owned());
    }
    for (index, (name, schema)) in before.iter().enumerate() {
        let occurrence = before[..index].iter().filter(|(other, _)| other == name).count();
        if let Some(schema) = schema
            && let Some((_, Some(actual))) = after.iter().filter(|(other, _)| other == name).nth(occurrence)
            && let Some(position) = schema_order(schema, actual, &format!("body.tools[{index}].schema"))
        {
            return Some(position);
        }
    }
    None
}

/// Borrow schemas without interpreting or rebuilding provider requests.
fn tool_schemas(value: &Value) -> Vec<(&str, Option<&Value>)> {
    let mut schemas = Vec::new();
    if let Some(tools) = value
        .get("tools")
        .or_else(|| value.pointer("/toolConfig/tools"))
        .and_then(Value::as_array)
    {
        for tool in tools {
            if let Some(declarations) = tool.get("functionDeclarations").and_then(Value::as_array) {
                for declaration in declarations {
                    push_tool_schema(declaration, &mut schemas);
                }
            } else {
                push_tool_schema(
                    tool.get("function").or_else(|| tool.get("toolSpec")).unwrap_or(tool),
                    &mut schemas,
                );
            }
        }
    }
    schemas
}

/// Select the explicitly named schema fields each provider envelope exposes.
fn push_tool_schema<'a>(tool: &'a Value, schemas: &mut Vec<(&'a str, Option<&'a Value>)>) {
    if let Some(name) = tool.get("name").and_then(Value::as_str) {
        let schema = tool
            .get("parameters")
            .or_else(|| tool.get("parametersJsonSchema"))
            .or_else(|| tool.get("input_schema"))
            .or_else(|| tool.pointer("/inputSchema/json"));
        schemas.push((name, schema));
    }
}

/// Schema value conversions may change types, but retained property names
/// must not be sorted or otherwise permuted as a side effect.
fn schema_order(before: &Value, after: &Value, position: &str) -> Option<String> {
    if let Some(position) = retained_order(before, after, position, true) {
        return Some(position);
    }
    match (before, after) {
        (Value::Object(before), Value::Object(after)) => {
            if schema_properties_reordered(before, after) {
                return Some(format!("{position}.properties"));
            }
            for (index, (key, before)) in before.iter().enumerate() {
                if let Some(after) = after.get(key)
                    && let Some(position) = schema_order(before, after, &format!("{position}.<key[{index}]>"))
                {
                    return Some(position);
                }
            }
        },
        (Value::Array(before), Value::Array(after)) => {
            for (index, (before, after)) in before.iter().zip(after).enumerate() {
                if let Some(position) = schema_order(before, after, &format!("{position}[{index}]")) {
                    return Some(position);
                }
            }
        },
        _ => {},
    }
    None
}

/// Compare property order even if a provider converted schema type values.
fn schema_properties_reordered(before: &Map<String, Value>, after: &Map<String, Value>) -> bool {
    match (before.get("properties"), after.get("properties")) {
        (Some(Value::Object(before)), Some(Value::Object(after))) => shared_keys_reordered(before, after),
        _ => false,
    }
}

/// Added, removed, and modified members cannot hide a permutation of retained keys.
fn shared_keys_reordered(before: &Map<String, Value>, after: &Map<String, Value>) -> bool {
    !before
        .keys()
        .filter(|key| after.contains_key(*key))
        .eq(after.keys().filter(|key| before.contains_key(*key)))
}

/// Validate JSON pointer escapes without accepting malformed `~` sequences.
fn valid_pointer(pointer: &str) -> bool {
    let mut chars = pointer.chars();
    while let Some(character) = chars.next() {
        if character == '~' && !matches!(chars.next(), Some('0' | '1')) {
            return false;
        }
    }
    true
}

/// Append one escaped object key to a JSON pointer.
fn child_pointer(parent: &str, key: &str) -> String {
    format!("{parent}/{}", key.replace('~', "~0").replace('/', "~1"))
}

/// Compare every unowned position, including member presence and order.
fn compare_unchanged(before: &Value, after: &Value, changes: &[&str], pointer: &str, position: &str) {
    if changes.contains(&pointer) {
        return;
    }
    match (before, after) {
        (Value::Object(before), Value::Object(after)) => {
            compare_object_members(before, after, changes, pointer, position);
        },
        (Value::Array(before), Value::Array(after)) => {
            assert_eq!(
                before.len(),
                after.len(),
                "undeclared JSON array length change at {position}"
            );
            for (index, (before, after)) in before.iter().zip(after).enumerate() {
                compare_unchanged(
                    before,
                    after,
                    changes,
                    &format!("{pointer}/{index}"),
                    &format!("{position}[{index}]"),
                );
            }
        },
        _ => assert!(before == after, "undeclared JSON value/type change at {position}"),
    }
}

/// Compare unowned object members in their original insertion order.
fn compare_object_members(
    before: &Map<String, Value>,
    after: &Map<String, Value>,
    changes: &[&str],
    pointer: &str,
    position: &str,
) {
    let before = before
        .iter()
        .filter(|(key, _)| !changes.contains(&child_pointer(pointer, key).as_str()));
    let mut after = after
        .iter()
        .filter(|(key, _)| !changes.contains(&child_pointer(pointer, key).as_str()));
    for (index, (key, value)) in before.enumerate() {
        let (actual_key, actual) = after.next().expect("undeclared JSON field removal");
        assert!(
            key == actual_key,
            "undeclared JSON field/order change at {position}.<key[{index}]>"
        );
        compare_unchanged(
            value,
            actual,
            changes,
            &child_pointer(pointer, key),
            &format!("{position}.<key[{index}]>"),
        );
    }
    assert!(after.next().is_none(), "undeclared JSON field addition at {position}");
}

/// An edited member is owned by the transformation; its siblings still retain order.
fn retained_members_reordered(before: &Map<String, Value>, after: &Map<String, Value>) -> bool {
    let before_keys = before
        .iter()
        .filter_map(|(key, value)| (after.get(key) == Some(value)).then_some(key));
    let after_keys = after
        .iter()
        .filter_map(|(key, value)| (before.get(key) == Some(value)).then_some(key));
    !before_keys.eq(after_keys)
}

/// Inspect retained content even when a surrounding object was transformed.
fn retained_order(before: &Value, after: &Value, position: &str, preserve_envelope: bool) -> Option<String> {
    match (before, after) {
        (Value::Object(before_object), Value::Object(after_object)) => {
            if (preserve_envelope || before == after) && retained_members_reordered(before_object, after_object) {
                return Some(position.to_owned());
            }
            for (index, (key, before)) in before_object.iter().enumerate() {
                if let Some(after) = after_object.get(key)
                    && let Some(mismatch) =
                        retained_order(before, after, &format!("{position}.<key[{index}]>"), preserve_envelope)
                {
                    return Some(mismatch);
                }
            }
        },
        (Value::Array(before), Value::Array(after)) => {
            for (index, (before, after)) in before.iter().zip(after).enumerate() {
                if let Some(mismatch) =
                    retained_order(before, after, &format!("{position}[{index}]"), preserve_envelope)
                {
                    return Some(mismatch);
                }
            }
        },
        _ => {},
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_changes_preserve_nested_keys_arrays_and_unknown_fields() {
        assert_json_changes_only(
            br#"{ "model":"alias", "tools":[{"z":1,"a":2}], "extra":{"b":1,"a":2} }"#,
            br#"{"model":"target","tools":[{"z":1,"a":2}],"extra":{"b":1,"a":2},"stream_options":{"include_usage":true}}"#,
            &["/model", "/stream_options"],
        );
    }

    #[test]
    fn declared_removal_and_escaped_nested_pointer_are_supported() {
        assert_json_changes_only(
            br#"{"model":"alias","a/b":{"~key":false,"keep":1},"last":2}"#,
            br#"{"a/b":{"~key":true,"keep":1},"last":2}"#,
            &["/model", "/a~1b/~0key"],
        );
    }

    #[test]
    #[should_panic(expected = "field/order change")]
    fn mutation_check_rejects_nested_key_sorting() {
        assert_json_changes_only(
            br#"{"model":"alias","tools":[{"z":1,"a":2}]}"#,
            br#"{"model":"target","tools":[{"a":2,"z":1}]}"#,
            &["/model"],
        );
    }

    #[test]
    #[should_panic(expected = "value/type change")]
    fn mutation_check_rejects_tool_reordering() {
        assert_json_changes_only(
            br#"{"model":"alias","tools":[{"name":"z"},{"name":"a"}]}"#,
            br#"{"model":"target","tools":[{"name":"a"},{"name":"z"}]}"#,
            &["/model"],
        );
    }

    #[test]
    #[should_panic(expected = "field removal")]
    fn mutation_check_rejects_unrelated_field_loss() {
        assert_json_changes_only(br#"{"model":"alias","extra":1}"#, br#"{"model":"target"}"#, &["/model"]);
    }

    #[test]
    #[should_panic(expected = "no-op path")]
    fn no_op_check_rejects_whitespace_normalization() {
        assert_body_unchanged(br#"{ "model": "alias" }"#, br#"{"model":"alias"}"#);
    }

    #[test]
    fn retained_order_check_runs_inside_a_modified_envelope() {
        let before: Value = serde_json::from_str(r#"{"model":"alias","tools":[{"z":1,"a":2}]}"#).unwrap();
        let after: Value = serde_json::from_str(r#"{"model":"target","tools":[{"a":2,"z":1}]}"#).unwrap();
        assert_eq!(
            retained_json_order_mismatch(&before, &after).as_deref(),
            Some("body.<key[1]>[0]")
        );
        assert!(retained_json_order_mismatch(&before, &before).is_none());
    }

    #[test]
    fn changed_siblings_do_not_hide_retained_key_reordering() {
        let before: Value = serde_json::from_str(r#"{"model":"alias","a":1,"b":2}"#).unwrap();
        let after: Value = serde_json::from_str(r#"{"model":"target","b":2,"a":1}"#).unwrap();
        assert_eq!(retained_json_order_mismatch(&before, &after).as_deref(), Some("body"));
        let after: Value = serde_json::from_str(r#"{"added":0,"a":1,"b":2}"#).unwrap();
        assert!(retained_json_order_mismatch(&before, &after).is_none());
        let after: Value = serde_json::from_str(r#"{"b":2,"a":1,"added":0}"#).unwrap();
        assert!(retained_json_order_mismatch(&before, &after).is_some());
    }

    #[test]
    fn translated_envelopes_keep_payload_order_checks() {
        let before: Value = serde_json::from_str(r#"{"model":"m","max_tokens":8,"metadata":{"z":1,"a":2}}"#).unwrap();
        let after: Value =
            serde_json::from_str(r#"{"max_tokens":8,"model":"target","metadata":{"z":1,"a":2}}"#).unwrap();
        assert!(translated_json_order_mismatch(&before, &after).is_none());
        let after: Value =
            serde_json::from_str(r#"{"max_tokens":8,"model":"target","metadata":{"a":2,"z":1}}"#).unwrap();
        assert!(translated_json_order_mismatch(&before, &after).is_some());
    }

    #[test]
    fn changed_schema_properties_do_not_hide_retained_key_reordering() {
        let before: Value =
            serde_json::from_str(r#"{"properties":{"z":{"type":"string"},"a":{"type":"number"},"removed":{}}}"#)
                .unwrap();
        let after: Value =
            serde_json::from_str(r#"{"properties":{"a":{"type":"NUMBER"},"z":{"type":"STRING"},"added":{}}}"#).unwrap();
        assert!(schema_properties_reordered(
            before.as_object().unwrap(),
            after.as_object().unwrap()
        ));
        let after: Value =
            serde_json::from_str(r#"{"properties":{"added":{},"z":{"type":"STRING"},"a":{"type":"NUMBER"}}}"#).unwrap();
        assert!(!schema_properties_reordered(
            before.as_object().unwrap(),
            after.as_object().unwrap()
        ));
    }

    #[test]
    fn translated_tool_wrappers_do_not_hide_schema_order_regressions() {
        let before: Value = serde_json::from_str(r#"{"tools":[{"name":"inspect","input_schema":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}]}"#).unwrap();
        let after: Value = serde_json::from_str(r#"{"tools":[{"function":{"name":"inspect","parameters":{"type":"object","properties":{"a":{"type":"number"},"z":{"type":"string"}}}}}]}"#).unwrap();
        assert!(retained_tool_order_mismatch(&before, &after).is_some());
        let after: Value = serde_json::from_str(r#"{"toolConfig":{"tools":[{"toolSpec":{"name":"inspect","inputSchema":{"json":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}}}]}}"#).unwrap();
        assert!(retained_tool_order_mismatch(&before, &after).is_none());
    }

    #[test]
    fn provider_schema_type_conversion_cannot_hide_property_sorting() {
        let before: Value = serde_json::from_str(r#"{"tools":[{"name":"inspect","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}]}"#).unwrap();
        let after: Value = serde_json::from_str(r#"{"tools":[{"functionDeclarations":[{"name":"inspect","parameters":{"type":"OBJECT","properties":{"a":{"type":"NUMBER"},"z":{"type":"STRING"}}}}]}]}"#).unwrap();
        assert_eq!(
            retained_tool_order_mismatch(&before, &after).as_deref(),
            Some("body.tools[0].schema.properties")
        );
    }

    #[test]
    fn duplicate_tool_names_match_the_corresponding_schema_occurrence() {
        let before: Value = serde_json::from_str(r#"{"tools":[{"name":"inspect","parameters":{"z":1,"a":2}},{"name":"inspect","parameters":{"a":2,"z":1}}]}"#).unwrap();
        assert!(retained_tool_order_mismatch(&before, &before).is_none());
        let after: Value = serde_json::from_str(r#"{"tools":[{"name":"inspect","parameters":{"z":1,"a":2}},{"name":"inspect","parameters":{"z":1,"a":2}}]}"#).unwrap();
        assert_eq!(
            retained_tool_order_mismatch(&before, &after).as_deref(),
            Some("body.tools[1].schema")
        );
    }

    #[test]
    fn tools_without_schemas_still_retain_their_relative_order() {
        let before: Value =
            serde_json::from_str(r#"{"tools":[{"function":{"name":"z"}},{"function":{"name":"a"}}]}"#).unwrap();
        let after: Value =
            serde_json::from_str(r#"{"tools":[{"function":{"name":"a"}},{"function":{"name":"z"}}]}"#).unwrap();
        assert!(retained_tool_order_mismatch(&before, &before).is_none());
        assert_eq!(
            retained_tool_order_mismatch(&before, &after).as_deref(),
            Some("body.tools")
        );
    }
}
