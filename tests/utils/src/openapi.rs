// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Semantic `OpenAPI` helpers shared by development tooling and integration tests.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{MapAccess, SeqAccess, Visitor},
    ser::{SerializeMap as _, SerializeSeq as _},
};

/// One semantic YAML value.
///
/// Unlike the generic YAML value, this preserves integer schema bounds outside
/// `i64`, as used by the pinned OpenAI document.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// YAML null.
    Null,
    /// YAML boolean.
    Bool(bool),
    /// Signed integer.
    Signed(i128),
    /// Unsigned integer.
    Unsigned(u128),
    /// Floating-point number.
    Float(f64),
    /// String scalar.
    String(String),
    /// Sequence.
    Sequence(Vec<Self>),
    /// Mapping.
    Mapping(Mapping),
}

impl Value {
    /// Borrow this value as a mapping.
    pub const fn as_mapping(&self) -> Option<&Mapping> {
        if let Self::Mapping(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Mutably borrow this value as a mapping.
    pub const fn as_mapping_mut(&mut self) -> Option<&mut Mapping> {
        if let Self::Mapping(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Borrow this value as a sequence.
    pub fn as_sequence(&self) -> Option<&[Self]> {
        if let Self::Sequence(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Borrow this value as a string.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

/// Ordered YAML mapping.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mapping(Vec<(Value, Value)>);

impl Mapping {
    /// Create an empty mapping.
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// Insert or replace one mapping entry.
    pub fn insert(&mut self, key: Value, value: Value) {
        if let Some((_existing_key, existing_value)) = self.0.iter_mut().find(|(candidate, _value)| *candidate == key) {
            *existing_value = value;
        } else {
            self.0.push((key, value));
        }
    }

    /// Return whether the mapping contains a key.
    pub fn contains_key(&self, key: &Value) -> bool {
        self.0.iter().any(|(candidate, _value)| candidate == key)
    }

    /// Borrow one mapping value.
    pub fn get(&self, key: &Value) -> Option<&Value> {
        self.0
            .iter()
            .find(|(candidate, _value)| candidate == key)
            .map(|(_key, value)| value)
    }

    /// Mutably borrow one mapping value.
    pub fn get_mut(&mut self, key: &Value) -> Option<&mut Value> {
        self.0
            .iter_mut()
            .find(|(candidate, _value)| candidate == key)
            .map(|(_key, value)| value)
    }

    /// Iterate over mapping entries in source order.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.0.iter().map(|(key, value)| (key, value))
    }

    /// Iterate over mapping keys.
    pub fn keys(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().map(|(key, _value)| key)
    }

    /// Iterate over mapping values.
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().map(|(_key, value)| value)
    }

    /// Return whether the mapping has no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<(Value, Value)> for Mapping {
    fn from_iter<T: IntoIterator<Item = (Value, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// Parse a complete `OpenAPI` YAML document without truncating wide integers.
///
/// # Errors
///
/// Returns an error when the YAML is malformed or contains duplicate mapping keys.
pub fn parse_document(source: &str) -> Result<Value, String> {
    serde_yaml::from_str(source).map_err(|error| format!("failed to parse pinned OpenAI schema: {error}"))
}

/// Resolve root component schemas and their complete local-reference closure.
///
/// # Errors
///
/// Returns an error when `components.schemas`, a requested root, or a referenced
/// local component is missing, or when a `$ref` is malformed or non-local.
pub fn component_schema_closure<'a>(
    document: &'a Value,
    roots: &[&str],
) -> Result<BTreeMap<String, &'a Value>, String> {
    let schemas = document
        .as_mapping()
        .and_then(|root| yaml_get(root, "components"))
        .and_then(Value::as_mapping)
        .and_then(|components| yaml_get(components, "schemas"))
        .and_then(Value::as_mapping)
        .ok_or_else(|| "pinned OpenAI schema has no components.schemas object".to_owned())?;

    let mut pending = roots.iter().map(|root| (*root).to_owned()).collect::<BTreeSet<_>>();
    let mut selected = BTreeMap::new();
    while let Some(name) = pending.pop_first() {
        if selected.contains_key(&name) {
            continue;
        }
        let schema = schemas
            .get(&Value::String(name.clone()))
            .ok_or_else(|| format!("schema references missing component {name}"))?;
        collect_schema_references(schema, &mut pending)?;
        selected.insert(name, schema);
    }
    Ok(selected)
}

/// Add local component references found below one semantic YAML value.
fn collect_schema_references(value: &Value, references: &mut BTreeSet<String>) -> Result<(), String> {
    match value {
        Value::Sequence(values) => {
            for value in values {
                collect_schema_references(value, references)?;
            }
        },
        Value::Mapping(fields) => {
            for (name, value) in fields.iter() {
                if name.as_str() == Some("$ref") {
                    let reference = value
                        .as_str()
                        .ok_or_else(|| "schema contains a non-string $ref".to_owned())?;
                    let component = reference
                        .strip_prefix("#/components/schemas/")
                        .ok_or_else(|| format!("schema contains unsupported reference {reference}"))?;
                    references.insert(component.to_owned());
                } else {
                    collect_schema_references(value, references)?;
                }
            }
        },
        Value::Bool(_) | Value::Float(_) | Value::Null | Value::Signed(_) | Value::String(_) | Value::Unsigned(_) => {},
    }
    Ok(())
}

/// Borrow one string-keyed YAML mapping value.
fn yaml_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(&Value::String(key.to_owned()))
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor)
    }
}

/// Serde visitor that preserves the complete scalar range used by `OpenAPI`.
struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a YAML value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        Value::deserialize(deserializer)
    }

    fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
        Ok(Value::Signed(i128::from(v)))
    }

    fn visit_i128<E>(self, v: i128) -> Result<Self::Value, E> {
        Ok(Value::Signed(v))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
        Ok(Value::Unsigned(u128::from(v)))
    }

    fn visit_u128<E>(self, v: u128) -> Result<Self::Value, E> {
        Ok(Value::Unsigned(v))
    }

    fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
        Ok(Value::Float(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
        Ok(Value::String(v))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element()? {
            values.push(value);
        }
        Ok(Value::Sequence(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Mapping::new();
        while let Some((key, value)) = map.next_entry()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom("duplicate YAML mapping key"));
            }
            values.insert(key, value);
        }
        Ok(Value::Mapping(values))
    }
}

impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Signed(value) => serializer.serialize_i128(*value),
            Self::Unsigned(value) => serializer.serialize_u128(*value),
            Self::Float(value) => serializer.serialize_f64(*value),
            Self::String(value) => serializer.serialize_str(value),
            Self::Sequence(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(value)?;
                }
                sequence.end()
            },
            Self::Mapping(values) => {
                let mut mapping = serializer.serialize_map(Some(values.0.len()))?;
                for (key, value) in &values.0 {
                    mapping.serialize_entry(key, value)?;
                }
                mapping.end()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_signed_schema_bounds_outside_i64() {
        let value = parse_document("minimum: -9223372036854776000\n").unwrap();
        let rendered = serde_yaml::to_string(&value).unwrap();

        assert!(rendered.contains("-9223372036854776000"));
    }

    #[test]
    fn rejects_duplicate_mapping_keys() {
        let error = parse_document("key: first\nkey: second\n").unwrap_err();
        assert!(error.contains("duplicate YAML mapping key"));
    }

    #[test]
    fn resolves_transitive_component_references() {
        let document = parse_document(
            "components:\n  schemas:\n    Root:\n      $ref: '#/components/schemas/Child'\n    Child:\n      type: string\n",
        )
        .unwrap();
        let selected = component_schema_closure(&document, &["Root"]).unwrap();

        assert_eq!(
            selected.keys().map(String::as_str).collect::<Vec<_>>(),
            ["Child", "Root"]
        );
    }
}
