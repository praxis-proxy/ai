// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Header-value safety shared by A2A and MCP dynamic promotions.

/// Match Praxis's header-value rule without allocating a `HeaderValue`.
///
/// HTAB, printable bytes, and non-ASCII UTF-8 bytes are accepted. Other
/// control bytes and DEL are rejected.
pub(super) fn contains_control_chars(value: &str) -> bool {
    !value
        .bytes()
        .all(|byte| byte == b'\t' || (byte >= 0x20 && byte != 0x7F))
}

#[cfg(test)]
mod tests {
    use super::contains_control_chars;

    #[test]
    fn control_classification_matches_header_value() {
        for byte in 0_u8..=127 {
            let value = char::from(byte).to_string();
            assert_eq!(
                !contains_control_chars(&value),
                http::HeaderValue::from_str(&value).is_ok(),
                "byte 0x{byte:02x} must match HTTP header safety"
            );
        }
        for value in ["café", "🙂", "café\ttab", "bad\rvalue", "bad\nvalue"] {
            assert_eq!(
                !contains_control_chars(value),
                http::HeaderValue::from_str(value).is_ok(),
                "value {value:?} must match HTTP header safety"
            );
        }
    }
}
