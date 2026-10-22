// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared helper for embedding generated JSON inside an HTML `<script>` element.
//!
//! Both flow-visualizer generators ([`crate::flow_generator`] and
//! [`crate::visualize_config`]) inline order-preserving JSON into a `<script>`
//! block (`const X = {...};`). A raw `</script>` — or a `<!--` — appearing inside
//! a JSON string value would prematurely close the element or open an HTML
//! comment, corrupting the page. Escaping the three characters that can do this
//! to their `\uXXXX` JSON forms keeps the parsed value byte-identical while
//! making the text inert in HTML.

/// Escape the three characters that could prematurely close a `<script>` element
/// or open an HTML comment when JSON is embedded in HTML. Each maps to a JSON
/// string escape, so the parsed value is unchanged; outside JSON strings these
/// characters never occur, so the JSON structure is untouched.
pub(crate) fn escape_json_for_script(json: &str) -> String {
    json.replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn neutralizes_script_close_and_comment_open() {
        let escaped = escape_json_for_script(r#""a</script><!--&b""#);
        assert!(!escaped.contains("</script>"), "closing tag is neutralized");
        assert!(!escaped.contains("<!--"), "comment open is neutralized");
        assert!(
            escaped.contains("\\u003c") && escaped.contains("\\u003e") && escaped.contains("\\u0026"),
            "the three HTML-significant characters are JSON-escaped"
        );
    }

    #[test]
    fn leaves_ordinary_json_untouched() {
        let json = r#"{"a":1,"b":"text","c":[true,null]}"#;
        assert_eq!(escape_json_for_script(json), json, "no < > & means no change");
    }

    #[test]
    fn escaped_output_still_parses_to_the_same_value() {
        let json = r#"{"note":"a < b && c > d"}"#;
        let escaped = escape_json_for_script(json);
        // Reversing the escapes recovers the original text losslessly.
        let restored = escaped
            .replace("\\u003c", "<")
            .replace("\\u003e", ">")
            .replace("\\u0026", "&");
        assert_eq!(restored, json, "escaping is losslessly reversible");
    }
}
