// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI response adapter for the core request limiter.


use praxis_filter::{FilterError, Rejection};
use utoipa::openapi::{
    RefOr, Schema,
    schema::{ObjectBuilder, Type},
};

/// Describe a computed retry delay in integer seconds.
pub(super) fn retry_after_schema() -> RefOr<Schema> {
    ObjectBuilder::new()
        .schema_type(Type::Integer)
        .minimum(Some(1_f64))
        .build()
        .into()
}

/// Add the OpenAI envelope while preserving the core limiter's status and headers.
pub(super) fn format_rejection(rejection: Rejection) -> Result<Rejection, FilterError> {
    let body = crate::openai::error::ErrorResponse::new(
        "Rate limit exceeded. Please retry after the indicated delay.",
        "rate_limit_error",
        Some("rate_limit_exceeded"),
        None,
    );
    let bytes = serde_json::to_vec(&body)
        .map_err(|error| FilterError::from(format!("openai_conversations: serialize rate limit error: {error}")))?;
    Ok(rejection
        .with_header("content-type", "application/json")
        .with_body(bytes))
}
