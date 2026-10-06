// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Pure helpers for walking Responses input content parts.
//!
//! Shared by `openai_file_resolve` (which downloads referenced files) and
//! `openai_doc_extract` (which only decodes inline data), so the extraction
//! filter does not depend on the network-fetching resolver.

/// Return the immutable content parts array for a given input item, if applicable.
pub(crate) fn content_parts(item: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    match item.get("type").and_then(serde_json::Value::as_str) {
        Some("message") => item.get("content").and_then(serde_json::Value::as_array),
        Some("function_call_output") => item.get("output").and_then(serde_json::Value::as_array),
        Some(_) => None,
        None => {
            if item.get("role").and_then(serde_json::Value::as_str).is_some() && item.get("content").is_some() {
                item.get("content").and_then(serde_json::Value::as_array)
            } else {
                None
            }
        },
    }
}

/// Return the mutable content parts array for a given input item,
/// if applicable.
pub(crate) fn content_parts_mut(item: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    match item.get("type").and_then(serde_json::Value::as_str) {
        Some("message") => item.get_mut("content").and_then(serde_json::Value::as_array_mut),
        Some("function_call_output") => item.get_mut("output").and_then(serde_json::Value::as_array_mut),
        Some(_) => None,
        None => {
            if item.get("role").and_then(serde_json::Value::as_str).is_some() && item.get("content").is_some() {
                item.get_mut("content").and_then(serde_json::Value::as_array_mut)
            } else {
                None
            }
        },
    }
}

/// Filename extensions and their inferred MIME types.
const KNOWN_EXTENSIONS: &[(&str, &str)] = &[
    ("csv", "text/csv"),
    ("doc", "application/msword"),
    (
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ("gif", "image/gif"),
    ("html", "text/html"),
    ("htm", "text/html"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("json", "application/json"),
    ("pdf", "application/pdf"),
    ("png", "image/png"),
    (
        "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ("txt", "text/plain"),
    ("webp", "image/webp"),
    (
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    ("xml", "application/xml"),
];

/// Infer MIME type from a filename extension.
pub(crate) fn infer_mime_from_filename(filename: Option<&str>) -> Option<&'static str> {
    let ext = filename?.rsplit('.').next()?;
    KNOWN_EXTENSIONS
        .iter()
        .find_map(|&(known, mime)| ext.eq_ignore_ascii_case(known).then_some(mime))
}

#[cfg(test)]
mod tests {
    use super::infer_mime_from_filename;

    #[test]
    fn mime_extension_matching_preserves_suffix_and_case_behavior() {
        for (name, expected) in [
            (Some("report.PdF"), Some("application/pdf")),
            (Some("archive.report.JpEg"), Some("image/jpeg")),
            (Some("report.unknown"), None),
            (Some("report.pñg"), None),
            (Some("report"), None),
            (Some("png"), Some("image/png")),
            (Some("report."), None),
            (None, None),
        ] {
            assert_eq!(infer_mime_from_filename(name), expected, "filename: {name:?}");
        }
    }

    #[test]
    fn mime_extension_matching_allocates_no_lowercase_copy() {
        let allocations = allocation_counter::measure(|| {
            std::hint::black_box(infer_mime_from_filename(Some("report.PdF")));
            std::hint::black_box(infer_mime_from_filename(Some("report.pñg")));
        });
        assert_eq!(
            allocations.count_total, 0,
            "MIME extension matching allocated: {allocations:?}"
        );
    }
}
