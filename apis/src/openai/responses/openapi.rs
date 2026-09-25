// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Generated `OpenAPI` description for Praxis-transformed Responses operations.

use utoipa::openapi::OpenApi;

use super::routes::operation_specs;
use crate::openai::operation;

/// Generate the local Responses implementation `OpenAPI` document as pretty JSON.
///
/// # Errors
///
/// Returns an error if generated schemas cannot be assembled or serialized.
pub fn implementation_openapi_json() -> Result<String, String> {
    let document = implementation_openapi_value()?;
    serde_json::to_string_pretty(&document).map_err(|error| format!("failed to serialize Responses OpenAPI: {error}"))
}

/// Build the local Responses implementation document from the operation
/// registry and its bound runtime contract types.
fn implementation_openapi() -> OpenApi {
    operation::implementation_openapi(
        "Praxis AI OpenAI Responses implementation",
        "0.1.0",
        "Responses",
        operation_specs().iter().map(|spec| &spec.definition),
    )
}

/// Convert the generated `OpenAPI` document to JSON value.
fn implementation_openapi_value() -> Result<serde_json::Value, String> {
    serde_json::to_value(implementation_openapi())
        .map_err(|error| format!("failed to serialize Responses OpenAPI: {error}"))
}
