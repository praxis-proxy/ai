// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Read the top-level request `model` while refusing a repeated one.
//!
//! JSON leaves repeated object keys undefined. `serde_json::Value` keeps the
//! last occurrence, other parsers keep the first or reject the document, and a
//! path that forwards the original bytes hands both occurrences to the
//! upstream. A routing decision taken from one occurrence could therefore
//! disagree with the model the backend serves, so body-based model routing
//! counts top-level `model` members before any parser collapses them into a
//! map. Occurrences are counted, not compared: an identical repeat is still
//! refused. Nested `model` keys and every other field are left to the upstream.
//!
//! Two entry points share that rule. [`parse_object`] replaces a
//! `serde_json::from_slice::<Value>` call for callers that need the parsed
//! object anyway, so the count costs no extra pass. [`scan`] serves callers
//! that only need the model and borrows it without allocating a JSON tree.
//!
//! ```
//! use praxis_ai_apis::model_field::{ModelScan, ParseError, parse_object, scan};
//!
//! assert!(matches!(scan(br#"{"model":"a"}"#), ModelScan::One(_)));
//! assert!(matches!(
//!     scan(br#"{"model":"a","model":"a"}"#),
//!     ModelScan::Duplicate
//! ));
//! assert!(matches!(
//!     parse_object(br#"{"model":"a","model":"b"}"#),
//!     Err(ParseError::DuplicateModel)
//! ));
//! assert!(parse_object(br#"{"model":"a","input":{"model":"b"}}"#).is_ok());
//! ```

use std::{borrow::Cow, fmt};

use serde::{
    Deserialize, Deserializer as _,
    de::{IgnoredAny, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Value, value::RawValue};

/// Top-level member name whose repetition makes routing ambiguous.
const MODEL: &str = "model";

/// Client-facing reason text shared by every rejecting adapter.
pub const DUPLICATE_MODEL_MESSAGE: &str = "request body contains more than one top-level 'model' field";

// -----------------------------------------------------------------------------
// Parsed Object
// -----------------------------------------------------------------------------

/// Why a request body could not be returned as a single-model JSON object.
#[derive(Debug)]
pub enum ParseError {
    /// The body is not one valid JSON document.
    Json(serde_json::Error),
    /// The body is valid JSON whose top-level value is not an object.
    NotObject,
    /// The top-level object repeats `model`.
    DuplicateModel,
}

/// Parse a request body into its top-level object, refusing a repeated `model`.
///
/// Builds the same map `serde_json::from_slice::<Value>` would, including
/// trailing-content rejection, so callers swap it in without a second pass.
///
/// # Errors
///
/// Returns [`ParseError`] for invalid JSON, a non-object document, or a
/// repeated top-level `model`.
pub fn parse_object(raw: &[u8]) -> Result<Map<String, Value>, ParseError> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let root = (&mut deserializer)
        .deserialize_any(RootVisitor)
        .map_err(ParseError::Json)?;
    deserializer.end().map_err(ParseError::Json)?;
    match root {
        Root::Object {
            object,
            duplicate_model: false,
        } => Ok(object),
        Root::Object {
            duplicate_model: true, ..
        } => Err(ParseError::DuplicateModel),
        Root::NotObject => Err(ParseError::NotObject),
    }
}

/// Top-level JSON value, kept only when it is an object.
enum Root {
    /// The members, plus whether `model` occurred more than once.
    Object {
        /// Members with `Value`'s last-wins insertion.
        object: Map<String, Value>,
        /// Whether `model` occurred more than once.
        duplicate_model: bool,
    },
    /// Any other JSON value; its contents are skipped.
    NotObject,
}

/// Visitor that builds a top-level object and counts its `model` members.
struct RootVisitor;

impl<'de> Visitor<'de> for RootVisitor {
    type Value = Root;

    /// Describe the accepted input when serde encounters malformed JSON.
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    /// Accept a non-object scalar so callers can tell it from invalid JSON.
    fn visit_bool<E>(self, _value: bool) -> Result<Root, E> {
        Ok(Root::NotObject)
    }

    /// Accept a non-object scalar so callers can tell it from invalid JSON.
    fn visit_i64<E>(self, _value: i64) -> Result<Root, E> {
        Ok(Root::NotObject)
    }

    /// Accept a non-object scalar so callers can tell it from invalid JSON.
    fn visit_u64<E>(self, _value: u64) -> Result<Root, E> {
        Ok(Root::NotObject)
    }

    /// Accept a non-object scalar so callers can tell it from invalid JSON.
    fn visit_f64<E>(self, _value: f64) -> Result<Root, E> {
        Ok(Root::NotObject)
    }

    /// Accept a non-object scalar so callers can tell it from invalid JSON.
    fn visit_str<E>(self, _value: &str) -> Result<Root, E> {
        Ok(Root::NotObject)
    }

    /// Accept `null` so callers can tell it from invalid JSON.
    fn visit_unit<E>(self) -> Result<Root, E> {
        Ok(Root::NotObject)
    }

    /// Skip a top-level array without allocating its elements.
    fn visit_seq<Access: SeqAccess<'de>>(self, mut seq: Access) -> Result<Root, Access::Error> {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(Root::NotObject)
    }

    /// Collect members as `Value` does while counting decoded `model` keys.
    fn visit_map<Access: MapAccess<'de>>(self, mut map: Access) -> Result<Root, Access::Error> {
        let mut object = Map::new();
        let mut seen_model = false;
        let mut duplicate_model = false;
        while let Some(key) = map.next_key::<String>()? {
            if key == MODEL {
                duplicate_model |= seen_model;
                seen_model = true;
            }
            let value = map.next_value::<Value>()?;
            object.insert(key, value);
        }
        Ok(Root::Object {
            object,
            duplicate_model,
        })
    }
}

// -----------------------------------------------------------------------------
// Borrowed Scan
// -----------------------------------------------------------------------------

/// Outcome of scanning a body for its top-level `model`.
#[derive(Debug)]
pub enum ModelScan<'a> {
    /// The body is not one valid JSON object.
    Invalid,
    /// The object has no top-level `model`.
    Absent,
    /// The object has exactly one top-level `model`, borrowed raw so its JSON
    /// type is not interpreted here.
    One(&'a RawValue),
    /// The object repeats `model`.
    Duplicate,
}

impl<'a> ModelScan<'a> {
    /// Decode a single string model, borrowing it unless it contains escapes.
    pub fn as_str(&self) -> Option<Cow<'a, str>> {
        match *self {
            Self::One(raw) => serde_json::from_str::<ModelString<'a>>(raw.get())
                .ok()
                .map(|model| model.0),
            Self::Invalid | Self::Absent | Self::Duplicate => None,
        }
    }
}

/// Scan a body for its top-level `model` without allocating its JSON tree.
///
/// Every member is consumed and trailing content is rejected, so a single
/// model is reported only for a complete document. Skipped values are not
/// fully validated; a caller that acts on the result must still parse the
/// body before trusting its other fields.
pub fn scan(raw: &[u8]) -> ModelScan<'_> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let Ok(scan) = (&mut deserializer).deserialize_map(ScanVisitor) else {
        return ModelScan::Invalid;
    };
    if deserializer.end().is_err() {
        return ModelScan::Invalid;
    }
    scan
}

/// Recognized root member; serde decodes escaped keys before matching.
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum Field {
    /// Client-facing model identity.
    Model,
    /// Every field whose value can be skipped.
    #[serde(other)]
    Other,
}

/// String decoder that borrows plain model names and owns escaped names.
#[derive(Deserialize)]
#[serde(transparent)]
struct ModelString<'a>(#[serde(borrow)] Cow<'a, str>);

/// Visitor that counts root `model` members and keeps the raw value of one.
struct ScanVisitor;

impl<'de> Visitor<'de> for ScanVisitor {
    type Value = ModelScan<'de>;

    /// Describe the required root object when serde encounters another JSON type.
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON request object")
    }

    /// Consume every member so a repeat later in the object is still counted.
    fn visit_map<Access: MapAccess<'de>>(self, mut map: Access) -> Result<Self::Value, Access::Error> {
        let mut model = None;
        let mut duplicate = false;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Model => duplicate |= model.replace(map.next_value::<&RawValue>()?).is_some(),
                Field::Other => {
                    map.next_value::<IgnoredAny>()?;
                },
            }
        }
        if duplicate {
            return Ok(ModelScan::Duplicate);
        }
        Ok(model.map_or(ModelScan::Absent, ModelScan::One))
    }
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
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    /// Bodies whose top-level `model` occurs more than once.
    const DUPLICATES: &[&[u8]] = &[
        br#"{"model":"a","model":"b"}"#,
        br#"{"model":"a","model":"a"}"#,
        br#"{"model":"a","messages":[],"model":"a"}"#,
        br#"{"model":null,"model":"a"}"#,
        br#"{"model":"a","model":42}"#,
        br#"{"model":"a","m\u006fdel":"b"}"#,
        br#"{"model":"a","model":"b","model":"c"}"#,
    ];

    #[test]
    fn parse_object_refuses_every_repeated_top_level_model() {
        for raw in DUPLICATES {
            assert!(
                matches!(parse_object(raw), Err(ParseError::DuplicateModel)),
                "repeat must be refused: {}",
                String::from_utf8_lossy(raw)
            );
        }
    }

    #[test]
    fn scan_refuses_every_repeated_top_level_model() {
        for raw in DUPLICATES {
            assert!(
                matches!(scan(raw), ModelScan::Duplicate),
                "repeat must be refused: {}",
                String::from_utf8_lossy(raw)
            );
        }
    }

    #[test]
    fn parse_object_matches_value_for_single_model() {
        let raw = br#"{"model":"gpt-4.1","input":{"model":"nested"},"tools":[{"model":"x"}],"n":1}"#;
        let object = parse_object(raw).unwrap();
        let expected: Value = serde_json::from_slice(raw).unwrap();
        assert_eq!(Value::Object(object), expected, "single-model parse must equal Value");
    }

    #[test]
    fn parse_object_keeps_last_wins_for_other_repeated_keys() {
        let object = parse_object(br#"{"model":"m","stream":true,"stream":false}"#).unwrap();
        assert_eq!(object["stream"], false, "only model is checked for repeats");
        assert_eq!(object.len(), 2, "repeated keys still collapse");
    }

    #[test]
    fn parse_object_separates_invalid_json_from_non_objects() {
        for raw in [&b""[..], b"{", br#"{"model":"a"} trailing"#, b"\xff"] {
            assert!(
                matches!(parse_object(raw), Err(ParseError::Json(_))),
                "invalid JSON: {}",
                String::from_utf8_lossy(raw)
            );
        }
        for raw in [
            &b"null"[..],
            b"true",
            b"1",
            b"-1",
            b"1.5",
            br#""s""#,
            br#"[{"model":"a"},{"model":"b"}]"#,
        ] {
            assert!(
                matches!(parse_object(raw), Err(ParseError::NotObject)),
                "non-object: {}",
                String::from_utf8_lossy(raw)
            );
        }
    }

    #[test]
    fn parse_object_rejects_trailing_content_after_a_duplicate() {
        assert!(
            matches!(
                parse_object(br#"{"model":"a","model":"b"} x"#),
                Err(ParseError::Json(_))
            ),
            "a document that is not valid JSON is reported as such"
        );
    }

    #[test]
    fn scan_reports_absent_single_and_invalid() {
        assert!(matches!(scan(br#"{"input":{"model":"x"}}"#), ModelScan::Absent));
        assert!(matches!(
            scan(br#"{"model":"plain","messages":[{"content":"long"}]}"#).as_str(),
            Some(Cow::Borrowed("plain"))
        ));
        assert_eq!(
            scan(br#"{"m\u006fdel":"pl\u0061in"}"#).as_str().as_deref(),
            Some("plain"),
            "escaped keys and values decode"
        );
        assert!(
            scan(br#"{"model":42}"#).as_str().is_none(),
            "a single non-string model is not a string"
        );
        for raw in [&b""[..], b"[]", br#"{"model":"a"} {}"#, b"{"] {
            assert!(
                matches!(scan(raw), ModelScan::Invalid),
                "invalid: {}",
                String::from_utf8_lossy(raw)
            );
        }
    }
}
