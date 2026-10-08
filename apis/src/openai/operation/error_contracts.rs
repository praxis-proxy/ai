// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared OpenAI error response contracts.
//!
//! The pinned OpenAI reference models its error responses with an
//! `ErrorResponse` body whose nested `Error` resource carries optional
//! misalignment details. Conversations and Responses operations reference
//! these shapes from their rate-limit and availability responses, so the
//! component schemas live here once and both families' generated
//! implementation documents register the same reference names.
//!
//! These types exist for contract generation only. Runtime error bodies are
//! produced by the owning filters, not by serializing these structs.

use std::borrow::Cow;

use utoipa::{
    PartialSchema, ToSchema,
    openapi::{
        Ref, RefOr,
        schema::{AnyOfBuilder, ObjectBuilder, Schema, Type},
    },
};

use super::{
    openapi::{HeaderSpec, MediaTypeSpec, ResponseSpec},
    schema_binding,
};

/// Media type shared by every OpenAI JSON error body.
const JSON_CONTENT_TYPE: &str = "application/json";

/// Message text for the misalignment continuation instruction.
const MISALIGNMENT_STEER_MESSAGE: &str = "The public continuation instruction.";

/// Message text for the misalignment block explanation.
const MISALIGNMENT_EXPLANATION: &str = "The public explanation for this block.";

/// Values the reference enumerates for the misalignment classification.
const MISALIGNMENT_ERROR_TYPE_VALUES: [&str; 4] = [
    "potentially_unintended_data_transfer",
    "potentially_unintended_data_access",
    "potentially_unintended_destructive_activity",
    "other",
];

/// Top-level error envelope returned by every OpenAI error response.
pub(crate) struct ErrorResponse;

impl PartialSchema for ErrorResponse {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(Type::Object)
                .property("error", Ref::from_schema_name("Error"))
                .required("error")
                .build(),
        ))
    }
}

impl ToSchema for ErrorResponse {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("ErrorResponse")
    }

    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        ErrorResource::schemas(schemas);
        schemas.push((Self::name().into(), Self::schema()));
    }
}

/// Nested error resource carried inside [`ErrorResponse`].
///
/// Named `Error` in the generated document; the Rust name avoids colliding
/// with `std::error::Error` conventions in scope.
pub(crate) struct ErrorResource;

impl ErrorResource {
    /// Schema for fields the reference models as a string or `null`.
    fn nullable_string() -> RefOr<Schema> {
        RefOr::T(Schema::AnyOf(
            AnyOfBuilder::new()
                .item(Schema::Object(ObjectBuilder::new().schema_type(Type::String).build()))
                .item(Schema::Object(ObjectBuilder::new().schema_type(Type::Null).build()))
                .build(),
        ))
    }
}

#[expect(
    clippy::large_stack_frames,
    reason = "utoipa builders construct the full error schema in one expression"
)]
impl PartialSchema for ErrorResource {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(Type::Object)
                .property("code", Self::nullable_string())
                .property(
                    "message",
                    Schema::Object(ObjectBuilder::new().schema_type(Type::String).build()),
                )
                .property("param", Self::nullable_string())
                .property(
                    "type",
                    Schema::Object(ObjectBuilder::new().schema_type(Type::String).build()),
                )
                .property(
                    "misalignment",
                    Ref::from_schema_name("MisalignmentErrorDetailsResource"),
                )
                .required("type")
                .required("message")
                .required("param")
                .required("code")
                .build(),
        ))
    }
}

impl ToSchema for ErrorResource {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("Error")
    }

    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        MisalignmentErrorDetailsResource::schemas(schemas);
        schemas.push((Self::name().into(), Self::schema()));
    }
}

/// Optional misalignment details attached to an [`ErrorResource`].
pub(crate) struct MisalignmentErrorDetailsResource;

impl PartialSchema for MisalignmentErrorDetailsResource {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(Type::Object)
                .property("error_type", Ref::from_schema_name("_MisalignmentErrorType"))
                .property(
                    "detailed_explanation",
                    Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(Type::String)
                            .description(Some(MISALIGNMENT_EXPLANATION))
                            .build(),
                    ),
                )
                .property("steer", Ref::from_schema_name("_MisalignmentSteer"))
                .build(),
        ))
    }
}

impl ToSchema for MisalignmentErrorDetailsResource {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("MisalignmentErrorDetailsResource")
    }

    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        MisalignmentErrorType::schemas(schemas);
        MisalignmentSteer::schemas(schemas);
        schemas.push((Self::name().into(), Self::schema()));
    }
}

/// Misalignment classification union: any string, with known values enumerated.
///
/// Named `_MisalignmentErrorType` in the generated document to match the
/// reference component.
pub(crate) struct MisalignmentErrorType;

impl PartialSchema for MisalignmentErrorType {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::AnyOf(
            AnyOfBuilder::new()
                .item(Schema::Object(ObjectBuilder::new().schema_type(Type::String).build()))
                .item(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(Type::String)
                        .enum_values(Some(MISALIGNMENT_ERROR_TYPE_VALUES))
                        .build(),
                ))
                .build(),
        ))
    }
}

impl ToSchema for MisalignmentErrorType {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("_MisalignmentErrorType")
    }

    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        schemas.push((Self::name().into(), Self::schema()));
    }
}

/// Public continuation instruction attached to misalignment details.
///
/// Named `_MisalignmentSteer` in the generated document to match the
/// reference component.
pub(crate) struct MisalignmentSteer;

impl PartialSchema for MisalignmentSteer {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(Type::Object)
                .property(
                    "message",
                    Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(Type::String)
                            .description(Some(MISALIGNMENT_STEER_MESSAGE))
                            .build(),
                    ),
                )
                .required("message")
                .build(),
        ))
    }
}

impl ToSchema for MisalignmentSteer {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("_MisalignmentSteer")
    }

    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        schemas.push((Self::name().into(), Self::schema()));
    }
}

/// Optional `Retry-After` header declared by rate-limit responses.
///
/// The schema is inline, matching the reference: an integer with a minimum
/// of one second.
#[cfg(feature = "openai-conversations")]
const RATE_LIMIT_RETRY_AFTER_HEADER: HeaderSpec = HeaderSpec::new(
    "Retry-After",
    "The minimum number of seconds to wait before retrying. This header is returned when the server has computed a retry delay and may be omitted for 429 responses that require user action.",
    || {
        Schema::Object(
            ObjectBuilder::new()
                .schema_type(Type::Integer)
                .minimum(Some(1.0))
                .build(),
        )
        .into()
    },
);

/// `Retry-After` header used by inference rate-limit and availability
/// responses.
#[cfg(feature = "openai-responses-openapi")]
const INFERENCE_RETRY_AFTER_HEADER: HeaderSpec = HeaderSpec::new(
    "Retry-After",
    "The minimum number of seconds to wait before retrying. This header is returned when the server has computed a retry delay and may be omitted.",
    || {
        Schema::Object(
            ObjectBuilder::new()
                .schema_type(Type::Integer)
                .minimum(Some(1.0))
                .build(),
        )
        .into()
    },
);

/// The `TooManyRequests` response referenced by all eight Conversations
/// operations in the pinned reference.
#[cfg(feature = "openai-conversations")]
pub(crate) const TOO_MANY_REQUESTS_RESPONSE: ResponseSpec = ResponseSpec {
    status: "429",
    description: "The request was rejected because a rate limit was exceeded.",
    content: &[MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!(ErrorResponse))],
    headers: &[RATE_LIMIT_RETRY_AFTER_HEADER],
};

/// The `InferenceRateLimited` response referenced by `POST /responses`.
#[cfg(feature = "openai-responses-openapi")]
pub(crate) const INFERENCE_RATE_LIMITED_RESPONSE: ResponseSpec = ResponseSpec {
    status: "429",
    description: "The request was rejected because a rate limit was exceeded. A slow_down error means traffic increased too quickly; reduce your request rate, then increase it gradually.",
    content: &[MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!(ErrorResponse))],
    headers: &[INFERENCE_RETRY_AFTER_HEADER],
};

/// The `InferenceServiceUnavailable` response referenced by `POST /responses`.
#[cfg(feature = "openai-responses-openapi")]
pub(crate) const INFERENCE_SERVICE_UNAVAILABLE_RESPONSE: ResponseSpec = ResponseSpec {
    status: "503",
    description: "The service is temporarily unavailable. A server_is_overloaded error means the requested model is temporarily overloaded; retry after a brief delay.",
    content: &[MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!(ErrorResponse))],
    headers: &[INFERENCE_RETRY_AFTER_HEADER],
};

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::indexing_slicing, clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::Value as JsonValue;

    use super::*;

    /// Component name an `ErrorResponse` chain must produce.
    const ERROR_RESOURCE: &str = "Error";

    #[test]
    fn error_response_schema_matches_reference_shape() {
        let schema = serde_json::to_value(ErrorResponse::schema()).unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["error"]["$ref"], "#/components/schemas/Error");
        assert_eq!(schema["required"], serde_json::json!(["error"]));
    }

    #[test]
    fn error_resource_schema_matches_reference_shape() {
        let schema = serde_json::to_value(ErrorResource::schema()).unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(
            schema["properties"]["code"],
            serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null"}]})
        );
        assert_eq!(schema["properties"]["message"], serde_json::json!({"type": "string"}));
        assert_eq!(
            schema["properties"]["param"],
            serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null"}]})
        );
        assert_eq!(schema["properties"]["type"], serde_json::json!({"type": "string"}));
        assert_eq!(
            schema["properties"]["misalignment"]["$ref"],
            "#/components/schemas/MisalignmentErrorDetailsResource"
        );
        assert_eq!(
            schema["required"],
            serde_json::json!(["type", "message", "param", "code"])
        );
    }

    #[test]
    fn misalignment_schemas_match_reference_shape() {
        let details = serde_json::to_value(MisalignmentErrorDetailsResource::schema()).unwrap();
        assert_eq!(details["type"], "object");
        assert_eq!(
            details["properties"]["error_type"]["$ref"],
            "#/components/schemas/_MisalignmentErrorType"
        );
        assert_eq!(details["properties"]["detailed_explanation"]["type"], "string");
        assert_eq!(
            details["properties"]["steer"]["$ref"],
            "#/components/schemas/_MisalignmentSteer"
        );

        let error_type = serde_json::to_value(MisalignmentErrorType::schema()).unwrap();
        assert_eq!(
            error_type["anyOf"][1]["enum"],
            serde_json::json!([
                "potentially_unintended_data_transfer",
                "potentially_unintended_data_access",
                "potentially_unintended_destructive_activity",
                "other"
            ])
        );

        let steer = serde_json::to_value(MisalignmentSteer::schema()).unwrap();
        assert_eq!(steer["type"], "object");
        assert_eq!(steer["properties"]["message"]["type"], "string");
        assert_eq!(steer["required"], serde_json::json!(["message"]));
    }

    #[test]
    fn error_response_chain_registers_every_referenced_component() {
        let mut schemas = Vec::new();
        ErrorResponse::schemas(&mut schemas);
        let names: Vec<&str> = schemas.iter().map(|(name, _)| name.as_str()).collect();
        for expected in [
            "MisalignmentErrorDetailsResource",
            "_MisalignmentErrorType",
            "_MisalignmentSteer",
            ERROR_RESOURCE,
            "ErrorResponse",
        ] {
            assert!(names.contains(&expected), "missing component {expected} in {names:?}");
        }
    }

    #[cfg(feature = "openai-conversations")]
    #[test]
    fn conversations_rate_limit_response_carries_error_body_and_retry_header() {
        assert_eq!(TOO_MANY_REQUESTS_RESPONSE.status, "429");
        assert_eq!(TOO_MANY_REQUESTS_RESPONSE.content.len(), 1);
        assert_eq!(TOO_MANY_REQUESTS_RESPONSE.content[0].content_type, JSON_CONTENT_TYPE);
        let headers = TOO_MANY_REQUESTS_RESPONSE.headers;
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].name, "Retry-After");
        let header_schema = serde_json::to_value(headers[0].schema()).unwrap();
        assert_eq!(header_schema["type"], "integer");
        assert_eq!(header_schema["minimum"], JsonValue::Number(1_u64.into()));
    }

    #[cfg(feature = "openai-responses-openapi")]
    #[test]
    fn responses_error_responses_declare_rate_limit_and_unavailable() {
        assert_eq!(INFERENCE_RATE_LIMITED_RESPONSE.status, "429");
        assert_eq!(INFERENCE_SERVICE_UNAVAILABLE_RESPONSE.status, "503");
        for response in [INFERENCE_RATE_LIMITED_RESPONSE, INFERENCE_SERVICE_UNAVAILABLE_RESPONSE] {
            assert_eq!(response.content.len(), 1);
            assert_eq!(response.headers.len(), 1);
            assert_eq!(response.headers[0].name, "Retry-After");
        }
    }
}
