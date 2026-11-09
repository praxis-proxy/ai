// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI response adapter for the core request limiter.

#![expect(
    clippy::large_stack_frames,
    reason = "utoipa schema builders allocate temporary stack space outside the request path"
)]

use praxis_filter::{FilterError, Rejection};
use serde::Serialize;
use utoipa::{
    ToSchema,
    openapi::{
        RefOr, Schema,
        schema::{AnyOfBuilder, ObjectBuilder, Type},
    },
};

/// JSON error envelope returned by Conversations admission.
#[derive(Serialize, ToSchema)]
pub(super) struct ErrorResponse {
    /// Structured error information.
    error: Error,
}

/// Pinned OpenAI error fields. Admission does not produce misalignment errors.
#[derive(Serialize, ToSchema)]
struct Error {
    /// Stable machine-readable code, or null when none applies.
    #[schema(required = true, schema_with = nullable_string)]
    code: Option<&'static str>,
    /// Human-readable failure explanation.
    message: &'static str,
    /// Request parameter implicated in the error, if any.
    #[schema(required = true, schema_with = nullable_string)]
    param: Option<&'static str>,
    /// OpenAI error category.
    #[serde(rename = "type")]
    error_type: &'static str,
    /// Optional policy-block details from the shared OpenAI error contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    misalignment: Option<MisalignmentErrorDetailsResource>,
}

/// Optional public details for a misalignment error.
#[derive(Serialize, ToSchema)]
struct MisalignmentErrorDetailsResource {
    /// Open classification of the policy failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(schema_with = misalignment_error_type)]
    error_type: Option<String>,
    /// Public explanation of the block.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    detailed_explanation: Option<String>,
    /// Public continuation instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    steer: Option<MisalignmentSteer>,
}

/// Public continuation instruction for a policy block.
#[derive(Serialize, ToSchema)]
struct MisalignmentSteer {
    /// Public instruction text.
    message: String,
}

/// Preserve the pin's string-or-null representation without narrowing it.
fn nullable_string() -> RefOr<Schema> {
    RefOr::T(Schema::AnyOf(
        AnyOfBuilder::new()
            .item(ObjectBuilder::new().schema_type(Type::String))
            .item(ObjectBuilder::new().schema_type(Type::Null))
            .build(),
    ))
}

/// Preserve the reference's open string classification with documented values.
fn misalignment_error_type() -> RefOr<Schema> {
    RefOr::T(Schema::AnyOf(
        AnyOfBuilder::new()
            .item(ObjectBuilder::new().schema_type(Type::String))
            .item(ObjectBuilder::new().schema_type(Type::String).enum_values(Some([
                "potentially_unintended_data_transfer",
                "potentially_unintended_data_access",
                "potentially_unintended_destructive_activity",
                "other",
            ])))
            .build(),
    ))
}

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
    let body = ErrorResponse {
        error: Error {
            code: Some("rate_limit_exceeded"),
            message: "Rate limit exceeded. Please retry after the indicated delay.",
            param: None,
            error_type: "rate_limit_error",
            misalignment: None,
        },
    };
    let bytes = serde_json::to_vec(&body)
        .map_err(|error| FilterError::from(format!("openai_conversations: serialize rate limit error: {error}")))?;
    Ok(rejection
        .with_header("content-type", "application/json")
        .with_body(bytes))
}
