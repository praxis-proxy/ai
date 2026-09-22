// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Pure helpers for walking Responses input content parts.
//!
//! Shared by `openai_file_resolve` (which downloads referenced files) and
//! `openai_doc_extract` (which only decodes inline data), so the extraction
//! filter does not depend on the network-fetching resolver.

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

/// Infer MIME type from a filename extension.
pub(crate) fn infer_mime_from_filename(filename: Option<&str>) -> Option<&'static str> {
    let ext = filename?.rsplit('.').next()?;
    match ext.to_ascii_lowercase().as_str() {
        "csv" => Some("text/csv"),
        "doc" => Some("application/msword"),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "gif" => Some("image/gif"),
        "html" | "htm" => Some("text/html"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "json" => Some("application/json"),
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        "txt" => Some("text/plain"),
        "webp" => Some("image/webp"),
        "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "xml" => Some("application/xml"),
        _ => None,
    }
}
