// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Borrow the last top-level model without allocating the request's JSON tree.

use std::{borrow::Cow, fmt};

use serde::{
    Deserialize, Deserializer as _,
    de::{IgnoredAny, MapAccess, Visitor},
};
use serde_json::value::RawValue;

/// Recognized root member; serde decodes escaped keys before matching.
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum Field {
    /// Client-facing model identity.
    Model,
    /// Every field whose value can be skipped during candidate selection.
    #[serde(other)]
    Other,
}

/// String decoder that borrows plain model names and owns escaped names.
#[derive(Deserialize)]
#[serde(transparent)]
struct Model<'a>(#[serde(borrow)] Cow<'a, str>);

/// Visitor retaining only the last model value, matching Value's duplicate keys.
struct ModelVisitor;

impl<'de> Visitor<'de> for ModelVisitor {
    type Value = Option<&'de RawValue>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON request object")
    }

    fn visit_map<Access: MapAccess<'de>>(self, mut map: Access) -> Result<Self::Value, Access::Error> {
        let mut model = None;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Model => model = Some(map.next_value::<&RawValue>()?),
                Field::Other => {
                    map.next_value::<IgnoredAny>()?;
                },
            }
        }
        Ok(model)
    }
}

/// Find a candidate model; matched requests still receive authoritative validation.
///
/// Scanning through the object preserves last-key precedence, including a final
/// non-string model, and rejects trailing JSON before a candidate is returned.
pub(super) fn peek(raw: &[u8]) -> Option<Cow<'_, str>> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let model = (&mut deserializer).deserialize_map(ModelVisitor).ok()??;
    deserializer.end().ok()?;
    serde_json::from_str::<Model<'_>>(model.get()).ok().map(|model| model.0)
}
