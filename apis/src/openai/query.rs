// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared query-string helpers for the `OpenAI` APIs.

use std::borrow::Cow;

use percent_encoding::percent_decode_str;

/// Strictly decode one query component, including form-style `+` spaces.
pub(crate) fn decode_query_component_strict(value: &str) -> Result<Cow<'_, str>, String> {
    if value.contains('+') {
        let normalized = value.replace('+', " ");
        return percent_decode_str(&normalized)
            .decode_utf8()
            .map(|decoded| Cow::Owned(decoded.into_owned()))
            .map_err(|e| format!("query parameter must be valid UTF-8: {e}"));
    }
    percent_decode_str(value)
        .decode_utf8()
        .map_err(|e| format!("query parameter must be valid UTF-8: {e}"))
}
