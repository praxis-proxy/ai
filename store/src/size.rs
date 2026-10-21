// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Measure compact JSON size against a ceiling without materializing a copy.

use std::io;

/// Whether the compact JSON serialization of `values` fits within `max_bytes`,
/// counted without allocating a serialized copy.
///
/// The writer subtracts each written chunk from the remaining allowance and
/// short-circuits the moment it would underflow, so an oversized tree stops
/// serialization early rather than building a full buffer first. Shared by the
/// in-memory backend's bounded reads and the SQL backends'
/// `get_conversation_bounded`, which measure a decoded record against a caller
/// limit after a cheaper raw-column preflight.
#[must_use]
pub fn json_values_fit(values: &[&serde_json::Value], max_bytes: usize) -> bool {
    struct Counter(usize);
    impl io::Write for Counter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0 = self
                .0
                .checked_sub(buf.len())
                .ok_or_else(|| io::Error::other("stored payload exceeds request limit"))?;
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut counter = Counter(max_bytes);
    values
        .iter()
        .all(|value| serde_json::to_writer(&mut counter, value).is_ok())
}

#[cfg(test)]
mod tests {
    use super::json_values_fit;

    #[test]
    fn empty_value_set_always_fits() {
        assert!(json_values_fit(&[], 0));
    }

    #[test]
    fn concatenated_values_are_summed_against_the_limit() {
        let a = serde_json::json!({"k": "vv"}); // {"k":"vv"} == 10 bytes
        let b = serde_json::json!([1, 2, 3]); // [1,2,3] == 7 bytes
        assert!(json_values_fit(&[&a, &b], 17), "17 bytes exactly fits 10 + 7");
        assert!(!json_values_fit(&[&a, &b], 16), "16 bytes rejects 10 + 7");
    }

    #[test]
    fn oversized_first_value_short_circuits() {
        let big = serde_json::Value::String("x".repeat(1_000));
        let small = serde_json::json!(1);
        assert!(!json_values_fit(&[&big, &small], 8));
    }
}
