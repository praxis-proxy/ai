// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conformance checks for the Files contract.
//!
//! Each `conformance_files_*` test ends with the sentinel line that
//! `cargo xtask openai-conformance` requires in its output.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{
    contracts::{DeleteFileResponse, FilePurpose, ListFilesResponse, OpenAiFile},
    implementation_openapi_json,
    routes::{FilesOperationSpec, operation_specs},
};
use crate::{
    openai::{
        error::ErrorResponse,
        operation::conformance::{
            OperationKey, assert_matches_schema, generated_operation_keys, generated_response_schemas,
            resolve_schema_ref, schema_matches_value,
        },
    },
    operation::Transport,
    operation_classifier::classify,
};

/// Generated Files document as JSON.
fn generated_openapi_spec() -> Value {
    serde_json::from_str(&implementation_openapi_json().unwrap()).unwrap()
}

/// Registry operation keys, sorted.
fn route_operation_keys() -> Vec<OperationKey> {
    let mut keys = operation_specs()
        .iter()
        .map(|spec| OperationKey::new(spec.method().as_str(), spec.spec_path))
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

/// Runtime path for `spec` with a fixed file ID.
fn runtime_path(spec: &FilesOperationSpec) -> String {
    spec.runtime_path().replace("{file_id}", "file_sync")
}

/// Serialize a DTO for schema validation.
fn payload(value: &impl serde::Serialize) -> Value {
    serde_json::to_value(value).unwrap()
}

/// Success payloads built through the DTO constructors, keyed by operation.
///
/// Handlers will replace these instances with real handler output.
fn success_payloads() -> BTreeMap<OperationKey, Vec<Value>> {
    let minimal = OpenAiFile::new("file_min", 3, 1_700_000_000, "a.txt", FilePurpose::UserData);
    let full = OpenAiFile::new("file_full", 10, 1_700_000_000, "b.jsonl", FilePurpose::Evals)
        .with_expires_at(1_700_003_600)
        .with_status_details("detail");

    let mut payloads = BTreeMap::new();
    payloads.insert(
        OperationKey::new("POST", "/files"),
        vec![payload(&minimal), payload(&full)],
    );
    payloads.insert(
        OperationKey::new("GET", "/files/{file_id}"),
        vec![payload(&minimal), payload(&full)],
    );
    payloads.insert(
        OperationKey::new("GET", "/files"),
        vec![
            payload(&ListFilesResponse::new(vec![minimal.clone(), full], true)),
            payload(&ListFilesResponse::new(Vec::new(), false)),
        ],
    );
    payloads.insert(
        OperationKey::new("DELETE", "/files/{file_id}"),
        vec![payload(&DeleteFileResponse::deleted("file_min"))],
    );
    payloads
}

#[test]
fn conformance_files_routes_match_runtime_registry() {
    let spec = generated_openapi_spec();
    assert_eq!(
        generated_operation_keys(&spec),
        route_operation_keys(),
        "generated OpenAPI paths must exactly match Files operation_specs()"
    );

    for operation in operation_specs() {
        let path = runtime_path(operation);
        let matched = classify(operation.method().as_str(), &path, Transport::Http)
            .unwrap_or_else(|| panic!("ai_operation did not classify {} {path}", operation.method().as_str()));
        assert_eq!(matched.operation_id, operation.operation_id());
        assert_eq!(
            matched.application_protocol.as_str(),
            "openai_files",
            "classifier registry identity drifted for {} {path}",
            operation.method().as_str()
        );
        assert_eq!(matched.request_body, operation.request_body());
        assert_eq!(
            matched.path_parameters.get(&path, "file_id"),
            operation.runtime_path().contains("{file_id}").then_some("file_sync"),
        );
    }
    println!("PRAXIS_CONFORMANCE_OK files route_dispatch");
}

#[test]
fn conformance_files_success_payloads_match_generated_response_schemas() {
    let spec = generated_openapi_spec();
    let schemas = generated_response_schemas(&spec, "application/json");
    let payloads = success_payloads();

    let schema_operations: Vec<_> = schemas.keys().collect();
    let payload_operations: Vec<_> = payloads.keys().collect();
    assert_eq!(
        payload_operations, schema_operations,
        "success payloads should cover every generated JSON response schema"
    );

    for (operation, schema) in &schemas {
        for (index, value) in payloads[operation].iter().enumerate() {
            assert_matches_schema(
                &spec,
                &format!("{} response #{index}", operation.label()),
                schema,
                value,
            );
        }
    }

    // The download returns raw bytes: no JSON payload exists to validate.
    let content = spec
        .pointer("/paths/~1files~1{file_id}~1content/get/responses/200/content")
        .and_then(Value::as_object)
        .unwrap();
    assert_eq!(content.keys().collect::<Vec<_>>(), ["application/octet-stream"]);
    let binary = resolve_schema_ref(&spec, &content["application/octet-stream"]["schema"]);
    assert_eq!(binary["type"], "string");
    assert_eq!(binary["format"], "binary");
    println!("PRAXIS_CONFORMANCE_OK files success_response_contract");
}

/// Error bodies with `code`/`param` null and set.
fn error_samples() -> [Value; 2] {
    [
        payload(&ErrorResponse::new("bad request", "invalid_request_error", None, None)),
        payload(&ErrorResponse::new(
            "no such file",
            "invalid_request_error",
            Some("file_not_found"),
            Some("file_id"),
        )),
    ]
}

/// Relevant portion of the pinned `OpenAPI` reference.
#[derive(serde::Deserialize)]
struct PinnedReference {
    /// Pinned paths by template and lowercase method.
    paths: BTreeMap<String, BTreeMap<String, Value>>,
}

#[test]
fn conformance_files_error_responses_match_pinned_contract() {
    let pinned: PinnedReference =
        serde_yaml::from_str(include_str!("../../../../docs/conformance/specs/openai-openapi.yaml")).unwrap();
    let spec = generated_openapi_spec();
    let samples = error_samples();

    for operation in operation_specs() {
        let method = operation.method().as_str().to_lowercase();
        let label = format!("{method} {}", operation.spec_path);
        let pinned_responses = pinned.paths[operation.spec_path][&method]["responses"]
            .as_object()
            .unwrap();
        let generated_responses = spec["paths"][operation.spec_path][&method]["responses"]
            .as_object()
            .unwrap();
        assert_eq!(
            pinned_responses.keys().collect::<Vec<_>>(),
            generated_responses.keys().collect::<Vec<_>>(),
            "{label}: generated statuses must equal the pinned statuses"
        );
        for (status, response) in generated_responses.iter().filter(|(status, _)| *status != "200") {
            let schema = &response["content"]["application/json"]["schema"];
            assert!(schema.is_object(), "{label} {status} must declare a JSON error schema");
            for value in &samples {
                assert_matches_schema(&spec, &format!("{label} {status}"), schema, value);
            }
        }
    }
    println!("PRAXIS_CONFORMANCE_OK files error_response_contract");
}

#[test]
fn conformance_files_generated_schema_check_rejects_invalid_payload() {
    let spec = generated_openapi_spec();
    let schema = spec
        .pointer("/components/schemas/OpenAIFile")
        .unwrap_or_else(|| panic!("generated OpenAPI missing OpenAIFile"));
    let valid = payload(&OpenAiFile::new("file_1", 3, 1, "a.txt", FilePurpose::UserData));
    assert!(schema_matches_value(&spec, schema, &valid), "baseline must match");

    let mut wrong_object = valid.clone();
    wrong_object["object"] = json!("wrong");
    assert!(
        !schema_matches_value(&spec, schema, &wrong_object),
        "schema check must reject a wrong object discriminator"
    );

    let mut missing_status = valid;
    missing_status.as_object_mut().unwrap().remove("status");
    assert!(
        !schema_matches_value(&spec, schema, &missing_status),
        "schema check must reject a file missing the required status"
    );
    println!("PRAXIS_CONFORMANCE_OK files schema_check_sensitivity");
}
