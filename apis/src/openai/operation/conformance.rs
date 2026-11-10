// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared conformance test helpers for operation families.
//!
//! A hand-written JSON-schema matcher over the generated `OpenAPI` document,
//! shared so each family validates its success payloads the same way.

use std::collections::BTreeMap;

use serde_json::Value;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct OperationKey {
    pub(crate) method: String,
    pub(crate) path: String,
}

impl OperationKey {
    pub(crate) fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
        }
    }

    pub(crate) fn label(&self) -> String {
        format!("{} {}", self.method, self.path)
    }
}

pub(crate) fn generated_operation_keys(spec: &Value) -> Vec<OperationKey> {
    let mut keys = Vec::new();
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("generated OpenAPI spec missing paths object"));

    for (path, path_item) in paths {
        let path_item = path_item
            .as_object()
            .unwrap_or_else(|| panic!("generated path item {path} should be an object"));
        for (method, _operation) in path_item {
            if matches!(method.as_str(), "delete" | "get" | "post") {
                keys.push(OperationKey::new(method.to_uppercase(), path));
            }
        }
    }

    keys.sort();
    keys
}

pub(crate) fn generated_response_schemas(spec: &Value, content_type: &str) -> BTreeMap<OperationKey, Value> {
    let mut schemas = BTreeMap::new();
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("generated OpenAPI spec missing paths object"));

    for (path, path_item) in paths {
        let path_item = path_item
            .as_object()
            .unwrap_or_else(|| panic!("generated path item {path} should be an object"));
        for (method, operation) in path_item {
            if !matches!(method.as_str(), "delete" | "get" | "post") {
                continue;
            }

            // Operations without this media type (e.g. a binary download) are
            // skipped; callers assert the expected operation set.
            let Some(schema) = operation.pointer(&format!(
                "/responses/200/content/{}/schema",
                content_type.replace('/', "~1")
            )) else {
                continue;
            };
            let previous = schemas.insert(OperationKey::new(method.to_uppercase(), path), schema.clone());
            assert!(previous.is_none(), "duplicate generated operation {method} {path}");
        }
    }

    schemas
}

pub(crate) fn assert_matches_schema(spec: &Value, path: &str, schema: &Value, value: &Value) {
    assert!(
        schema_matches_value(spec, schema, value),
        "{path} does not match generated schema: {value}"
    );
    let schema = resolve_schema_ref(spec, schema);

    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
        assert!(
            enum_values.contains(value),
            "{path} should be one of {enum_values:?}, got {value}"
        );
    }

    if let Some(schema_type) = schema.get("type").and_then(Value::as_str) {
        assert_schema_type(path, schema_type, value);
    } else if schema.get("properties").is_some() || schema.get("required").is_some() {
        assert_schema_type(path, "object", value);
    }

    if let Some(items_schema) = schema.get("items") {
        for (idx, item) in value
            .as_array()
            .unwrap_or_else(|| panic!("{path} should be an array"))
            .iter()
            .enumerate()
        {
            assert_matches_schema(spec, &format!("{path}[{idx}]"), items_schema, item);
        }
    }

    let Some(obj) = value.as_object() else {
        return;
    };

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for required_name in required {
            let required_name = required_name.as_str().unwrap();
            assert!(
                obj.contains_key(required_name),
                "{path} missing required property {required_name}"
            );
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (property_name, property_schema) in properties {
            if let Some(property_value) = obj.get(property_name) {
                assert_matches_schema(
                    spec,
                    &format!("{path}.{property_name}"),
                    property_schema,
                    property_value,
                );
            }
        }
    }

    if let Some(additional_schema) = schema.get("additionalProperties").filter(|schema| schema.is_object()) {
        let properties = schema.get("properties").and_then(Value::as_object);
        for (property_name, property_value) in obj {
            if properties.is_none_or(|properties| !properties.contains_key(property_name)) {
                assert_matches_schema(
                    spec,
                    &format!("{path}.{property_name}"),
                    additional_schema,
                    property_value,
                );
            }
        }
    }
}

pub(crate) fn schema_matches_value(spec: &Value, schema: &Value, value: &Value) -> bool {
    let schema = resolve_schema_ref(spec, schema);

    if value.is_null() && schema.get("nullable").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array)
        && variants
            .iter()
            .filter(|variant| schema_matches_value(spec, variant, value))
            .count()
            != 1
    {
        return false;
    }
    if let Some(variants) = schema.get("anyOf").and_then(Value::as_array)
        && !variants
            .iter()
            .any(|variant| schema_matches_value(spec, variant, value))
    {
        return false;
    }
    if let Some(variants) = schema.get("allOf").and_then(Value::as_array)
        && !variants
            .iter()
            .all(|variant| schema_matches_value(spec, variant, value))
    {
        return false;
    }
    if schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| !values.contains(value))
    {
        return false;
    }
    if let Some(schema_type) = schema.get("type") {
        let matches_type = schema_type.as_str().map_or_else(
            || {
                schema_type.as_array().is_some_and(|types| {
                    types
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|kind| value_has_type(value, kind))
                })
            },
            |kind| value_has_type(value, kind),
        );
        if !matches_type {
            return false;
        }
    }

    if let Some(items_schema) = schema.get("items") {
        let Some(items) = value.as_array() else {
            return false;
        };
        if !items.iter().all(|item| schema_matches_value(spec, items_schema, item)) {
            return false;
        }
    }

    let Some(object) = value.as_object() else {
        return schema.get("properties").is_none() && schema.get("required").is_none();
    };
    if schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .any(|name| !object.contains_key(name))
        })
    {
        return false;
    }

    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties
        && properties.iter().any(|(name, property_schema)| {
            object
                .get(name)
                .is_some_and(|property| !schema_matches_value(spec, property_schema, property))
        })
    {
        return false;
    }

    match schema.get("additionalProperties") {
        Some(Value::Bool(false)) => {
            properties.is_none_or(|properties| object.keys().all(|name| properties.contains_key(name)))
        },
        Some(additional_schema @ Value::Object(_)) => object.iter().all(|(name, property)| {
            properties.is_some_and(|properties| properties.contains_key(name))
                || schema_matches_value(spec, additional_schema, property)
        }),
        _ => true,
    }
}

pub(crate) fn resolve_schema_ref<'a>(spec: &'a Value, schema: &'a Value) -> &'a Value {
    let Some(ref_path) = schema.get("$ref").and_then(Value::as_str) else {
        return schema;
    };
    let pointer = ref_path.strip_prefix('#').unwrap_or(ref_path);
    spec.pointer(pointer)
        .unwrap_or_else(|| panic!("missing generated schema ref {ref_path}"))
}

pub(crate) fn assert_schema_type(path: &str, schema_type: &str, value: &Value) {
    let matches = value_has_type(value, schema_type);
    assert!(matches, "{path} should be {schema_type}, got {value}");
}

pub(crate) fn value_has_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.as_f64().is_some(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => false,
    }
}
