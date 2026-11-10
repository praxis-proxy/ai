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
    /// Opaque continuation target, absent for admission failures.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(schema_with = review_target_schema)]
    review_target: Option<String>,
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

/// Describe the pinned nullable continuation target, including its character bounds.
fn review_target_schema() -> RefOr<Schema> {
    RefOr::T(Schema::AnyOf(
        AnyOfBuilder::new()
            .item(
                ObjectBuilder::new()
                    .schema_type(Type::String)
                    .min_length(Some(1))
                    .max_length(Some(96))
                    .pattern(Some("^[A-Za-z0-9._~:-]+$"))
                    // Utoipa has no `not` builder. Its flattened extra fields
                    // preserve this JSON Schema keyword from the reference.
                    .extensions(Some(
                        [("not", serde_json::json!({"pattern": "[^A-Za-z0-9._~:-]"}))]
                            .into_iter()
                            .collect(),
                    )),
            )
            .item(ObjectBuilder::new().schema_type(Type::Null))
            .build(),
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_target_matches_pinned_schema_and_remains_optional() -> Result<(), Box<dyn std::error::Error>> {
        let expected = pinned_review_target()?;
        let generated = serde_json::to_value(<MisalignmentErrorDetailsResource as utoipa::PartialSchema>::schema())?;
        let target = generated
            .pointer("/properties/review_target")
            .ok_or("generated misalignment details have no review_target")?;
        assert_eq!(
            target, &expected,
            "the generated optional field must match the independent reference"
        );
        assert!(
            generated
                .get("required")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|fields| !fields.contains(&serde_json::json!("review_target"))),
            "review_target must remain optional"
        );
        Ok(())
    }

    #[test]
    fn review_target_enforces_nullable_character_and_length_bounds() -> Result<(), Box<dyn std::error::Error>> {
        let schema = serde_json::to_value(review_target_schema())?;
        let validator = jsonschema::validator_for(&schema)?;
        for value in [
            serde_json::Value::Null,
            serde_json::json!("review_1.a~b:c-d"),
            serde_json::json!("a".repeat(96)),
        ] {
            validator.validate(&value).map_err(|error| error.to_string())?;
        }
        for value in [
            String::new(),
            "a".repeat(97),
            "has space".to_owned(),
            "target\n".to_owned(),
            "é".to_owned(),
        ] {
            assert!(
                !validator.is_valid(&serde_json::json!(value)),
                "invalid continuation target: {value:?}"
            );
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    #[derive(serde::Deserialize)]
    /// Relevant portion of the independent upstream reference.
    struct Reference {
        components: Components,
    }
    #[derive(serde::Deserialize)]
    /// Reference component collection.
    struct Components {
        schemas: Schemas,
    }
    #[derive(serde::Deserialize)]
    /// Only the shared error detail schema is needed.
    struct Schemas {
        #[serde(rename = "MisalignmentErrorDetailsResource")]
        details: serde_json::Value,
    }

    /// Extract only the relevant schema so unrelated upstream numbers need not fit JSON integers.
    fn pinned_review_target() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let reference: Reference =
            serde_yaml::from_str(include_str!("../../../../docs/conformance/specs/openai-openapi.yaml"))?;
        let mut expected = reference
            .components
            .schemas
            .details
            .pointer("/properties/review_target")
            .ok_or("pinned misalignment details have no review_target")?
            .clone();
        expected
            .pointer_mut("/anyOf/0")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("pinned review target has no string branch")?
            .remove("description");
        Ok(expected)
    }
}
