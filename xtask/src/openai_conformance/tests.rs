// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::{
    Args,
    area::{CONFORMANCE_AREAS, CONVERSATIONS_SCOPE, OPENAI_REFERENCE_SPEC},
    coverage::calculate_coverage,
    find_missing_sentinels,
    json_report::report_json,
    model::{
        CoverageMode, OasdiffAreaDrift, OasdiffAreaReport, OasdiffOperationDrift, OperationKey, ReferenceSourceReport,
        RuntimeVerificationCheck, SpecOperation, SpecSourceReport, SupportedOperation, percent,
    },
    oasdiff::{
        OASDIFF_VERSION, build_oasdiff_report, collect_oasdiff_operation_status, operation_drift_from_diff,
        parse_oasdiff_version,
    },
    parse_percent,
    reference::build_reference_projection,
    selected_areas,
    spec::{parse_openapi_operations, read_spec, scope_operations},
};

/// Small representative `OpenAPI` document.
const SPEC: &str = "
openapi: 3.1.0
paths:
  /conversations:
    post:
      operationId: createConversation
      tags: [Conversations]
  /conversations?beta=true:
    post:
      operationId: createConversationBeta
      tags: [Conversations]
  /conversations/{conversation_id}:
    get:
      operationId: getConversation
      tags: [Conversations]
  /conversations/{conversation_id}/items:
    get:
      operationId: listConversationItems
      tags: [Conversations]
  /conversations/legacy:
    get:
      operationId: listLegacyConversations
      tags: [Conversations]
      deprecated: true
  /chat/completions:
    post:
      operationId: createChatCompletion
      tags: [Chat]
  /models:
    get:
      operationId: listModels
      tags: [Models]
";

fn scoped_operations() -> Vec<SpecOperation> {
    let operations = parse_openapi_operations(SPEC).unwrap();
    scope_operations(operations, CONVERSATIONS_SCOPE)
}

fn test_reference() -> ReferenceSourceReport {
    ReferenceSourceReport {
        source: "openai-test".to_owned(),
        source_sha256: "test-source-sha256".to_owned(),
        provenance: None,
    }
}

fn test_sources() -> Vec<SpecSourceReport> {
    vec![SpecSourceReport {
        area_id: "conversations",
        area: "Conversations",
        operations: 5,
        projection_sha256: "test-projection-sha256".to_owned(),
    }]
}

fn support(method: &str, path: &str, area: &str, mode: CoverageMode) -> SupportedOperation {
    SupportedOperation {
        method: method.to_owned(),
        path: path.to_owned(),
        area: area.to_owned(),
        mode,
        evidence: "test evidence".to_owned(),
    }
}

fn test_supported_operations() -> Vec<SupportedOperation> {
    vec![
        support("POST", "/conversations", "Conversations", CoverageMode::Local),
        support(
            "GET",
            "/conversations/{conversation_id}",
            "Conversations",
            CoverageMode::Local,
        ),
        support(
            "GET",
            "/conversations/{conversation_id}/items",
            "Conversations",
            CoverageMode::Local,
        ),
    ]
}

fn registered_supported_operations() -> Vec<SupportedOperation> {
    CONFORMANCE_AREAS
        .iter()
        .flat_map(|area| (area.supported_operations)())
        .collect()
}

fn test_args() -> Args {
    Args {
        openai_spec: "openai-test".to_owned(),
        areas: Vec::new(),
        include_deprecated: false,
        include_beta: false,
        list_covered: false,
        list_missing: false,
        implementation_spec: None,
        list_oasdiff: false,
        output_json: None,
        fail_under: None,
        fail_oasdiff_under: None,
    }
}

#[test]
fn parses_operations() {
    let operations = parse_openapi_operations(SPEC).unwrap();
    assert_eq!(operations.len(), 7, "should parse all operation entries");
    assert!(
        operations
            .iter()
            .any(|op| op.key == OperationKey::new("POST", "/conversations")),
        "POST /conversations should be parsed"
    );
    assert!(
        operations.iter().any(|op| op.deprecated),
        "deprecated marker should be parsed"
    );
    assert!(operations.iter().any(|op| op.beta), "beta path should be marked");
}

#[test]
fn scopes_only_conversations() {
    let operations = scoped_operations();

    assert_eq!(operations.len(), 5, "scope should exclude chat and model operations");
    assert!(
        operations.iter().all(|op| op.area == "Conversations"),
        "all scoped operations should receive an area"
    );
    assert!(
        operations
            .iter()
            .all(|op| !op.key.path.starts_with("/chat") && !op.key.path.starts_with("/models")),
        "unsupported API families should not enter the denominator"
    );
}

#[test]
fn data_driven_scope_matches_only_complete_path_segments() {
    let files = super::model::OperationScope::new("files", "Files", &["/files"]);
    assert!(files.matches("/files"));
    assert!(files.matches("/files/{file_id}/content"));
    assert!(!files.matches("/filesystem"));
}

#[test]
fn selects_registered_areas_by_stable_id() {
    let mut args = test_args();
    args.areas = vec!["conversations".to_owned()];
    let selected = selected_areas(&args).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected.first().unwrap().scope.id, "conversations");

    args.areas = vec!["files".to_owned()];
    assert!(selected_areas(&args).is_err(), "unregistered areas should be rejected");
}

#[test]
fn discovers_supported_operations_from_runtime_route_specs() {
    let supported = registered_supported_operations();
    let keys = supported.iter().map(SupportedOperation::key).collect::<BTreeSet<_>>();

    assert!(
        keys.contains(&OperationKey::new(
            "DELETE",
            "/conversations/{conversation_id}/items/{item_id}"
        )),
        "Conversations item delete support should come from runtime route specs"
    );
    assert!(
        supported
            .iter()
            .any(|operation| operation.evidence.contains("conversations_operation_specs")),
        "discovered operations should carry route-spec evidence"
    );
}

#[test]
fn rejects_remote_spec_sources() {
    let err = read_spec("https://example.com/openapi.yaml").unwrap_err();
    assert!(
        err.contains("remote OpenAPI spec sources are not supported"),
        "remote specs should be rejected: {err}"
    );
}

#[test]
fn excludes_deprecated_and_beta_by_default() {
    let report = calculate_coverage(
        test_reference(),
        test_sources(),
        scoped_operations(),
        &test_supported_operations(),
        false,
        false,
    );

    assert_eq!(report.ignored.deprecated, 1, "one deprecated operation ignored");
    assert_eq!(report.ignored.beta, 1, "one beta operation ignored");
    assert_eq!(
        report.considered.len(),
        3,
        "stable denominator should exclude ignored operations"
    );
    assert_eq!(
        report.covered.len(),
        3,
        "all stable scoped operations should be covered"
    );
    assert!(report.missing.is_empty(), "unscoped model list should not be missing");
}

#[test]
fn includes_deprecated_and_beta_when_requested() {
    let report = calculate_coverage(
        test_reference(),
        test_sources(),
        scoped_operations(),
        &test_supported_operations(),
        true,
        true,
    );

    assert_eq!(report.ignored.deprecated, 0, "deprecated should be included");
    assert_eq!(report.ignored.beta, 0, "beta should be included");
    assert_eq!(report.considered.len(), 5, "all scoped operations should be considered");
    assert_eq!(report.covered.len(), 3, "deprecated and beta entries are not claimed");
    assert_eq!(
        report.missing.len(),
        2,
        "deprecated and beta scoped entries should be missing"
    );
}

#[test]
fn coverage_percent_handles_empty_denominator() {
    let report = calculate_coverage(
        test_reference(),
        test_sources(),
        Vec::new(),
        &test_supported_operations(),
        false,
        false,
    );
    assert_eq!(report.coverage_percent(), 0.0, "empty denominator should be 0%");
}

#[test]
#[expect(clippy::too_many_lines, reason = "representative oasdiff JSON fixture")]
fn extracts_oasdiff_deleted_and_modified_operations() {
    let operations = scoped_operations();
    let diff = json!({
        "paths": {
            "deleted": ["/conversations/{conversation_id}"],
            "modified": {
                "/conversations": {
                    "operations": {
                        "modified": {
                            "POST": {
                                "requestBody": {
                                    "content": {
                                        "modified": {
                                            "application/json": {}
                                        }
                                    }
                                },
                                "responses": {
                                    "modified": {
                                        "200": {
                                            "content": {
                                                "modified": {
                                                    "application/json": {
                                                        "schema": {
                                                            "properties": {
                                                                "modified": {
                                                                    "id": {
                                                                        "type": {
                                                                            "added": ["integer"],
                                                                            "deleted": ["string"]
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    let mut missing = BTreeSet::new();
    let mut drifted = BTreeMap::new();
    let mut area_details = Vec::new();

    collect_oasdiff_operation_status(
        &diff,
        CONVERSATIONS_SCOPE,
        &operations,
        &mut missing,
        &mut drifted,
        &mut area_details,
    );

    assert!(
        missing.contains(&OperationKey::new("GET", "/conversations/{conversation_id}")),
        "deleted path should mark its scoped operations as missing"
    );
    let drift = drifted
        .get(&OperationKey::new("POST", "/conversations"))
        .expect("modified operation should be classified as drifted");
    assert!(drift.has_request_drift(), "requestBody drift should be classified");
    assert!(drift.has_response_drift(), "responses drift should be classified");
    assert!(
        drift
            .response_details
            .iter()
            .any(|detail| detail.contains("responses.200.content.application/json.schema.properties.id.type")),
        "response type drift should be included in details"
    );
}

#[test]
fn oasdiff_report_counts_missing_and_drifted_as_nonconformant() {
    let operations = scoped_operations();
    let missing = BTreeSet::from([OperationKey::new("GET", "/conversations/{conversation_id}")]);
    let drift = OasdiffOperationDrift {
        key: OperationKey::new("POST", "/conversations"),
        request_details: Vec::new(),
        response_details: vec!["responses.200.content.application/json.schema.type".to_owned()],
        other_details: Vec::new(),
    };
    let drifted = BTreeMap::from([(drift.key.clone(), drift)]);

    let report = build_oasdiff_report(OASDIFF_VERSION, &operations, missing, drifted, Vec::new(), Vec::new());

    assert_eq!(
        report.total,
        operations.len(),
        "all scoped operations should be counted"
    );
    assert_eq!(report.missing.len(), 1, "missing operation should be preserved");
    assert_eq!(report.drifted.len(), 1, "drifted operation should be preserved");
    assert_eq!(report.response_drift_count(), 1, "response drift should be counted");
    assert_eq!(report.request_drift_count(), 0, "request drift should not be counted");
    assert_eq!(
        report.conformant,
        operations.len() - 2,
        "missing and drifted operations should reduce exact conformance"
    );
}

#[test]
#[expect(clippy::too_many_lines, reason = "representative oasdiff JSON fixture")]
fn schema_property_named_description_is_not_ignored() {
    let drift = operation_drift_from_diff(
        OperationKey::new("POST", "/conversations"),
        &json!({
            "responses": {
                "modified": {
                    "200": {
                        "content": {
                            "modified": {
                                "application/json": {
                                    "schema": {
                                        "properties": {
                                            "modified": {
                                                "description": {
                                                    "type": {
                                                        "added": ["integer"],
                                                        "deleted": ["string"]
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }),
    );

    let drift = drift.expect("schema property drift should be retained");
    assert!(
        drift.response_details.iter().any(|detail| {
            detail == "responses.200.content.application/json.schema.properties.description.type.added"
        })
    );
}

#[test]
#[expect(clippy::too_many_lines, reason = "JSON report fixture and assertions")]
fn json_report_includes_oasdiff_missing_and_response_drift() {
    let operations = scoped_operations();
    let mut report = calculate_coverage(
        test_reference(),
        test_sources(),
        operations,
        &test_supported_operations(),
        false,
        false,
    );
    let missing = BTreeSet::from([OperationKey::new("GET", "/conversations/{conversation_id}")]);
    let drift = OasdiffOperationDrift {
        key: OperationKey::new("POST", "/conversations"),
        request_details: Vec::new(),
        response_details: vec!["responses.200.content.application/json.schema.type".to_owned()],
        other_details: Vec::new(),
    };
    let drifted = BTreeMap::from([(drift.key.clone(), drift)]);
    let area_report = OasdiffAreaReport {
        area_id: "conversations",
        area: "Conversations",
        implementation_source: "implementation-spec".to_owned(),
        total: report.considered.len(),
        conformant: report.considered.len().saturating_sub(missing.len() + drifted.len()),
        missing: missing.iter().cloned().collect(),
        drifted: drifted.values().cloned().collect(),
        inherited_details: vec!["security.deleted.ApiKeyAuth".to_owned()],
    };
    report.oasdiff = Some(build_oasdiff_report(
        OASDIFF_VERSION,
        &report.considered,
        missing,
        drifted,
        vec![OasdiffAreaDrift {
            area: "Conversations",
            details: vec!["security.deleted.ApiKeyAuth".to_owned()],
        }],
        vec![area_report],
    ));

    let value = report_json(&report, &test_args());

    assert_eq!(value.pointer("/schema_version").and_then(Value::as_u64), Some(3));
    assert!(value.get("overall_conformance").is_none());
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/operation_contracts/exact")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/operation_contracts/exact_percent")
            .and_then(Value::as_f64),
        Some(percent(1, report.considered.len()))
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/enabled")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/operation_contracts/missing")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/operation_contracts/response_drifted")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/fixes_required/summary/operations_to_fix")
            .and_then(Value::as_u64),
        Some(2)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/fixes_required/by_operation/0/fixes/0/kind")
            .and_then(Value::as_str),
        Some("missing_operation")
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/fixes_required/by_operation/1/fixes/0/kind")
            .and_then(Value::as_str),
        Some("response_schema")
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/areas/0/operation_contracts/drifted_operations/0/response_details/0")
            .and_then(Value::as_str),
        Some("responses.200.content.application/json.schema.type")
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/area_contracts/0/exact")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/areas/0/implementation_source")
            .and_then(Value::as_str),
        Some("implementation-spec")
    );
    assert_eq!(
        value
            .pointer("/capability_coverage/by_mode/3/operations")
            .and_then(Value::as_u64),
        Some(3)
    );
}

#[test]
fn json_report_explains_when_oasdiff_was_not_run() {
    let report = calculate_coverage(
        test_reference(),
        test_sources(),
        scoped_operations(),
        &test_supported_operations(),
        false,
        false,
    );
    let value = report_json(&report, &test_args());

    assert_eq!(
        value
            .pointer("/capability_coverage/coverage_percent")
            .and_then(Value::as_f64),
        Some(100.0)
    );
    assert_eq!(
        value
            .pointer("/owned_contract_conformance/enabled")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert!(
        value
            .pointer("/owned_contract_conformance/reason")
            .and_then(Value::as_str)
            .is_some_and(|reason| reason.contains("not generated")),
        "disabled owned-contract section should explain why it is absent"
    );
}

#[test]
fn coverage_mode_labels_and_owned_modes_are_stable() {
    assert_eq!(
        CoverageMode::ALL.map(CoverageMode::as_str),
        ["passthrough", "inspect", "transform", "local"]
    );
    assert!(!CoverageMode::Passthrough.is_owned());
    assert!(!CoverageMode::Inspect.is_owned());
    assert!(CoverageMode::Transform.is_owned());
    assert!(CoverageMode::Local.is_owned());
}

#[test]
fn finite_percentage_parser_rejects_nan_infinity_and_out_of_range_values() {
    assert_eq!(parse_percent("0").unwrap(), 0.0);
    assert_eq!(parse_percent("100").unwrap(), 100.0);
    for invalid in ["NaN", "inf", "-1", "100.1"] {
        assert!(parse_percent(invalid).is_err(), "{invalid} should be rejected");
    }
}

#[test]
fn parses_only_the_pinned_oasdiff_version_shape() {
    assert_eq!(
        parse_oasdiff_version("oasdiff version 1.23.0\n").unwrap(),
        OASDIFF_VERSION
    );
    assert!(parse_oasdiff_version("1.23.0").is_err());
}

#[test]
#[expect(clippy::too_many_lines, reason = "representative inherited drift fixture")]
fn reports_global_security_and_path_level_parameter_drift() {
    let operations = scoped_operations();
    let diff = json!({
        "security": {"deleted": ["ApiKeyAuth"]},
        "components": {"securitySchemes": {"deleted": ["ApiKeyAuth"]}},
        "paths": {"modified": {
            "/conversations/{conversation_id}": {
                "parameters": {"deleted": {"path": ["conversation_id"]}}
            }
        }}
    });
    let mut missing = BTreeSet::new();
    let mut drifted = BTreeMap::new();
    let mut area_details = Vec::new();

    collect_oasdiff_operation_status(
        &diff,
        CONVERSATIONS_SCOPE,
        &operations,
        &mut missing,
        &mut drifted,
        &mut area_details,
    );

    assert!(area_details.iter().any(|detail| detail.starts_with("security.")));
    assert!(
        area_details
            .iter()
            .any(|detail| detail.starts_with("components.securitySchemes."))
    );
    let get = drifted
        .get(&OperationKey::new("GET", "/conversations/{conversation_id}"))
        .expect("path-level parameter drift should apply to GET");
    assert!(get.has_request_drift());
}

#[test]
#[expect(clippy::too_many_lines, reason = "representative reference projection fixture")]
fn projects_conversations_with_inherited_security_and_transitive_components() {
    let source = b"
openapi: 3.1.0
info: {title: Test, version: 1.0.0}
security:
  - ApiKeyAuth: []
paths:
  /conversations/{conversation_id}:
    parameters:
      - {in: path, name: conversation_id, required: true, schema: {type: string}}
    get:
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema: {$ref: '#/components/schemas/Conversation'}
  /models:
    get:
      responses: {'200': {description: OK}}
components:
  securitySchemes:
    ApiKeyAuth: {type: http, scheme: bearer}
  schemas:
    Conversation:
      type: object
      properties:
        metadata: {$ref: '#/components/schemas/Metadata'}
    Metadata: {type: object}
    Unused: {type: string}
";

    let bundle = build_reference_projection(source, CONVERSATIONS_SCOPE).unwrap();
    let bundle: Value = serde_yaml::from_str(&bundle).unwrap();
    assert!(bundle.pointer("/security/0/ApiKeyAuth").is_some());
    assert!(bundle.pointer("/components/securitySchemes/ApiKeyAuth").is_some());
    assert!(bundle.pointer("/components/schemas/Conversation").is_some());
    assert!(bundle.pointer("/components/schemas/Metadata").is_some());
    assert!(bundle.pointer("/components/schemas/Unused").is_none());
    assert!(bundle.pointer("/paths/~1models").is_none());
    assert!(
        bundle
            .pointer("/paths/~1conversations~1{conversation_id}/parameters")
            .is_some()
    );
}

#[test]
fn reference_projection_rejects_external_refs() {
    let source = b"
openapi: 3.1.0
info: {title: Test, version: 1.0.0}
paths:
  /conversations:
    get:
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema: {$ref: 'http://127.0.0.1/schema.yaml'}
components: {}
";
    let error = build_reference_projection(source, CONVERSATIONS_SCOPE).unwrap_err();
    assert!(error.contains("external"), "unexpected error: {error}");
}

#[test]
fn complete_pinned_spec_projects_conversations_and_future_files_areas() {
    let source = read_spec(OPENAI_REFERENCE_SPEC).unwrap();
    let conversations = build_reference_projection(source.as_bytes(), CONVERSATIONS_SCOPE).unwrap();
    let files_scope = super::model::OperationScope::new("files", "Files", &["/files"]);
    let files = build_reference_projection(source.as_bytes(), files_scope).unwrap();

    assert_eq!(parse_openapi_operations(&conversations).unwrap().len(), 8);
    assert_eq!(parse_openapi_operations(&files).unwrap().len(), 5);
}

// -----------------------------------------------------------------------------
// Sentinel verification unit tests
// -----------------------------------------------------------------------------

const TEST_CHECKS: &[RuntimeVerificationCheck] = &[
    RuntimeVerificationCheck {
        kind: "alpha",
        evidence: "test::alpha",
        success_sentinel: "PRAXIS_CONFORMANCE_OK area alpha",
    },
    RuntimeVerificationCheck {
        kind: "beta",
        evidence: "test::beta",
        success_sentinel: "PRAXIS_CONFORMANCE_OK area beta",
    },
];

#[test]
fn sentinel_exact_line_match_passes() {
    let stdout = "other output\nPRAXIS_CONFORMANCE_OK area alpha\nPRAXIS_CONFORMANCE_OK area beta\n";
    assert!(find_missing_sentinels(stdout, TEST_CHECKS).is_empty());
}

#[test]
fn sentinel_missing_sentinel_detected() {
    let stdout = "other output\nPRAXIS_CONFORMANCE_OK area alpha\n";
    let missing = find_missing_sentinels(stdout, TEST_CHECKS);
    assert_eq!(missing, vec!["PRAXIS_CONFORMANCE_OK area beta"]);
}

#[test]
fn sentinel_partial_substring_not_accepted() {
    let stdout = "prefix PRAXIS_CONFORMANCE_OK area alpha suffix\nPRAXIS_CONFORMANCE_OK area beta\n";
    let missing = find_missing_sentinels(stdout, TEST_CHECKS);
    assert_eq!(missing, vec!["PRAXIS_CONFORMANCE_OK area alpha"]);
}

#[test]
fn sentinel_empty_output_reports_all_missing() {
    let missing = find_missing_sentinels("", TEST_CHECKS);
    assert_eq!(missing.len(), 2);
}

#[test]
fn sentinel_leading_trailing_whitespace_trimmed() {
    let stdout = "  PRAXIS_CONFORMANCE_OK area alpha  \n\tPRAXIS_CONFORMANCE_OK area beta\t\n";
    assert!(find_missing_sentinels(stdout, TEST_CHECKS).is_empty());
}
