// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Lossless-enough YAML tree for semantic OpenAPI projection.
//!
//! The upstream OpenAI document contains integer schema bounds outside `i64`.
//! The generic YAML value used elsewhere in the workspace cannot represent
//! those scalars, so this tree retains signed and unsigned 128-bit integers.

use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{MapAccess, SeqAccess, Visitor},
    ser::{SerializeMap as _, SerializeSeq as _},
};

/// One semantic YAML value.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    /// YAML null.
    Null,
    /// YAML boolean.
    Bool(bool),
    /// Signed integer, including `OpenAPI` bounds outside `i64`.
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
    pub(super) fn as_mapping(&self) -> Option<&Mapping> {
        if let Self::Mapping(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Mutably borrow this value as a mapping.
    pub(super) fn as_mapping_mut(&mut self) -> Option<&mut Mapping> {
        if let Self::Mapping(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Borrow this value as a sequence.
    pub(super) fn as_sequence(&self) -> Option<&[Self]> {
        if let Self::Sequence(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Borrow this value as a string.
    pub(super) fn as_str(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

/// Ordered YAML mapping.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Mapping(Vec<(Value, Value)>);

impl Mapping {
    /// Create an empty mapping.
    pub(super) const fn new() -> Self {
        Self(Vec::new())
    }

    /// Insert or replace one mapping entry.
    pub(super) fn insert(&mut self, key: Value, value: Value) {
        if let Some((_existing_key, existing_value)) = self.0.iter_mut().find(|(candidate, _value)| *candidate == key) {
            *existing_value = value;
        } else {
            self.0.push((key, value));
        }
    }

    /// Return whether the mapping contains a key.
    pub(super) fn contains_key(&self, key: &Value) -> bool {
        self.0.iter().any(|(candidate, _value)| candidate == key)
    }

    /// Borrow one mapping value.
    pub(super) fn get(&self, key: &Value) -> Option<&Value> {
        self.0
            .iter()
            .find(|(candidate, _value)| candidate == key)
            .map(|(_key, value)| value)
    }

    /// Mutably borrow one mapping value.
    pub(super) fn get_mut(&mut self, key: &Value) -> Option<&mut Value> {
        self.0
            .iter_mut()
            .find(|(candidate, _value)| candidate == key)
            .map(|(_key, value)| value)
    }

    /// Iterate over mapping entries in source order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.0.iter().map(|(key, value)| (key, value))
    }

    /// Iterate over mapping keys.
    pub(super) fn keys(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().map(|(key, _value)| key)
    }

    /// Iterate over mapping values.
    pub(super) fn values(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().map(|(_key, value)| value)
    }

    /// Return whether the mapping has no entries.
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<(Value, Value)> for Mapping {
    fn from_iter<T: IntoIterator<Item = (Value, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
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
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn preserves_signed_schema_bounds_outside_i64() {
        let source = "minimum: -9223372036854776000\n";
        let value: Value = serde_yaml::from_str(source).unwrap();
        let rendered = serde_yaml::to_string(&value).unwrap();

        assert!(rendered.contains("-9223372036854776000"));
    }

    #[test]
    fn rejects_duplicate_mapping_keys() {
        let error = serde_yaml::from_str::<Value>("key: first\nkey: second\n").unwrap_err();
        assert!(error.to_string().contains("duplicate YAML mapping key"));
    }
}
