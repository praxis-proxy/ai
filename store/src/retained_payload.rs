// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conservative accounting for JSON retained while history is rehydrated.
//!
//! Byte length accounts for payload copies; the node allowance accounts for
//! `serde_json::Value` containers, keys, and scalar values, whose per-value
//! overhead is high for sparse or deeply nested JSON. This is an admission
//! estimate, not a measurement of allocator RSS.

use std::io;

use crate::StoreError;

/// Accounting allowance per serialized payload byte.
pub const RETAINED_JSON_BYTE_MULTIPLIER: usize = 32;

/// Accounting allowance per JSON container, key, or scalar value.
pub const RETAINED_JSON_NODE_BYTES: usize = 8_192;

/// Measure the retained allowance for one raw JSON payload without building a
/// `serde_json::Value` tree. The caller can deserialize only after this
/// function succeeds.
///
/// # Errors
///
/// Returns [`StoreError::PayloadTooLarge`] when the payload exceeds `max_charge`
/// and [`StoreError::Serialization`] when its JSON syntax is malformed.
pub fn json_payload_charge(json: &[u8], max_charge: usize) -> Result<usize, StoreError> {
    let bytes_charge = json
        .len()
        .checked_mul(RETAINED_JSON_BYTE_MULTIPLIER)
        .ok_or(StoreError::PayloadTooLarge)?;
    if bytes_charge > max_charge {
        return Err(StoreError::PayloadTooLarge);
    }
    let node_allowance = (max_charge - bytes_charge) / RETAINED_JSON_NODE_BYTES;
    let nodes = count_json_nodes(json, node_allowance)?;
    let node_charge = nodes
        .checked_mul(RETAINED_JSON_NODE_BYTES)
        .ok_or(StoreError::PayloadTooLarge)?;
    bytes_charge
        .checked_add(node_charge)
        .filter(|charge| *charge <= max_charge)
        .ok_or(StoreError::PayloadTooLarge)
}

/// Measure already-decoded values before cloning them across a store boundary.
/// Serialized byte counting uses a writer, and structural counting walks the
/// borrowed trees, so this does not allocate another JSON payload.
///
/// `identifiers` are strings copied with the retained record. They are charged
/// as serialized string scalars, including their quotes.
///
/// # Errors
///
/// Returns [`StoreError::PayloadTooLarge`] when checked arithmetic overflows
/// or the calculated allowance exceeds `max_charge`.
pub fn json_values_charge<'a>(
    values: impl IntoIterator<Item = &'a serde_json::Value> + 'a,
    identifiers: &[&str],
    max_charge: usize,
) -> Result<usize, StoreError> {
    let mut bytes = ByteCounter(0);
    let mut nodes = 0_usize;
    for value in values {
        serde_json::to_writer(&mut bytes, value).map_err(|error| StoreError::Serialization(error.to_string()))?;
        nodes = nodes
            .checked_add(value_node_count(value).ok_or(StoreError::PayloadTooLarge)?)
            .ok_or(StoreError::PayloadTooLarge)?;
    }
    for identifier in identifiers {
        // Identifier strings are serialized JSON scalars at the read boundary.
        serde_json::to_writer(&mut bytes, identifier).map_err(|error| StoreError::Serialization(error.to_string()))?;
        nodes = nodes.checked_add(1).ok_or(StoreError::PayloadTooLarge)?;
    }
    retained_charge(bytes.0, nodes)
        .filter(|charge| *charge <= max_charge)
        .ok_or(StoreError::PayloadTooLarge)
}

/// Calculate the shared byte-plus-node charge with checked arithmetic.
pub fn retained_charge(serialized_bytes: usize, json_nodes: usize) -> Option<usize> {
    serialized_bytes
        .checked_mul(RETAINED_JSON_BYTE_MULTIPLIER)?
        .checked_add(json_nodes.checked_mul(RETAINED_JSON_NODE_BYTES)?)
}

/// Count the `serde_json::Value` containers, keys, and scalars in a decoded
/// tree. Mirrors the structural entries [`count_json_nodes`] counts in raw JSON.
fn value_node_count(value: &serde_json::Value) -> Option<usize> {
    use serde_json::Value;

    let children = match value {
        Value::Array(values) => values
            .iter()
            .try_fold(0_usize, |count, child| count.checked_add(value_node_count(child)?))?,
        Value::Object(values) => values.iter().try_fold(values.len(), |count, (_, child)| {
            count.checked_add(value_node_count(child)?)
        })?,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => 0,
    };
    children.checked_add(1)
}

/// Count JSON structural entries while respecting quoted strings. Full syntax
/// validation remains the responsibility of the deserializer that follows.
#[expect(
    clippy::too_many_lines,
    reason = "the lexical scan keeps string and number state in one loop"
)]
fn count_json_nodes(json: &[u8], max_nodes: usize) -> Result<usize, StoreError> {
    let mut nodes = 0_usize;
    let mut index = 0_usize;
    while let Some(&byte) = json.get(index) {
        match byte {
            b'{' | b'[' => {
                nodes = add_node(nodes, max_nodes)?;
                index += 1;
            },
            b'"' => {
                index = skip_string(json, index)?;
                // Object keys and string scalar values each correspond to one
                // retained entry. Both receive the same structural allowance.
                nodes = add_node(nodes, max_nodes)?;
            },
            b'-' | b'0'..=b'9' => {
                nodes = add_node(nodes, max_nodes)?;
                index += 1;
                while json
                    .get(index)
                    .is_some_and(|byte| byte.is_ascii_digit() || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-'))
                {
                    index += 1;
                }
            },
            b't' | b'f' | b'n' => {
                nodes = add_node(nodes, max_nodes)?;
                while json.get(index).is_some_and(u8::is_ascii_alphabetic) {
                    index += 1;
                }
            },
            // Whitespace, commas, colons, closers, and any other byte only
            // advance the cursor without creating a retained entry.
            _ => index += 1,
        }
    }
    Ok(nodes)
}

/// Return the index just past a quoted string beginning at `start`, honoring
/// backslash escapes.
///
/// # Errors
///
/// Returns [`StoreError::Serialization`] when the string is never terminated.
fn skip_string(json: &[u8], start: usize) -> Result<usize, StoreError> {
    let mut index = start.checked_add(1).ok_or(StoreError::PayloadTooLarge)?;
    while let Some(byte) = json.get(index) {
        match byte {
            b'"' => return Ok(index + 1),
            b'\\' => {
                index = index.checked_add(2).ok_or(StoreError::PayloadTooLarge)?;
            },
            _ => index += 1,
        }
    }
    Err(StoreError::Serialization("unterminated JSON string".to_owned()))
}

/// Increment the running node count, rejecting once it would exceed `max_nodes`.
///
/// # Errors
///
/// Returns [`StoreError::PayloadTooLarge`] on overflow or when the count passes
/// `max_nodes`.
fn add_node(nodes: usize, max_nodes: usize) -> Result<usize, StoreError> {
    let next = nodes.checked_add(1).ok_or(StoreError::PayloadTooLarge)?;
    if next > max_nodes {
        return Err(StoreError::PayloadTooLarge);
    }
    Ok(next)
}

/// Accumulates serialized JSON byte length without retaining the bytes.
struct ByteCounter(usize);

impl io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(buf.len())
            .ok_or_else(|| io::Error::other("serialized JSON size overflow"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn byte_and_structure_allowance_is_checked_before_value_construction() {
        let input = br#"{"a":[{"b":"text"},true,null,17]}"#;
        let expected_nodes = 1 /* root object */
            + 1 /* key a */
            + 1 /* array */
            + 1 /* object */
            + 1 /* key b */
            + 1 /* text */
            + 1 /* true */
            + 1 /* null */
            + 1; /* number */
        let expected = input.len() * RETAINED_JSON_BYTE_MULTIPLIER + expected_nodes * RETAINED_JSON_NODE_BYTES;
        assert_eq!(json_payload_charge(input, expected).unwrap(), expected);
        assert!(matches!(
            json_payload_charge(input, expected - 1),
            Err(StoreError::PayloadTooLarge)
        ));
    }

    #[test]
    fn strings_and_escapes_do_not_create_false_structural_nodes() {
        let input = br#"{"text":"{[true,\\\"x\"]}","n":1}"#;
        let charge = json_payload_charge(input, usize::MAX).unwrap();
        // root object + two keys + two values
        assert_eq!(
            charge,
            input.len() * RETAINED_JSON_BYTE_MULTIPLIER + 5 * RETAINED_JSON_NODE_BYTES
        );
        serde_json::from_slice::<serde_json::Value>(input).unwrap();
    }

    #[test]
    fn existing_values_are_measured_without_a_serialized_copy() {
        let value = serde_json::json!({"a": [1, 2]});
        let json = serde_json::to_vec(&value).unwrap();
        let raw_charge = json_payload_charge(&json, usize::MAX).unwrap();
        let value_charge = json_values_charge([&value], &[], usize::MAX).unwrap();
        assert_eq!(raw_charge, value_charge);
    }

    #[test]
    fn retained_history_fixture_fits_the_default_allowance() {
        let payload = format!("obsolete {}", "obsolete ".repeat(32_768));
        let json = serde_json::to_vec(&serde_json::json!({"text": payload})).unwrap();
        let charge = json_payload_charge(&json, 64 * 1024 * 1024).unwrap();
        assert!(charge < 64 * 1024 * 1024);
    }
}
