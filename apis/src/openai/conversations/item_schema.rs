// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Validation and `OpenAPI` components for Conversation item unions.

use std::{collections::BTreeMap, sync::LazyLock};

use jsonschema::{Draft, Validator};
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Generated item contracts derived from the pinned OpenAI document.
const ITEM_CONTRACTS_JSON: &str = include_str!("item_contracts.json");

/// Parsed item schemas shared by validation and `OpenAPI` generation.
static ITEM_SCHEMAS: LazyLock<Result<BTreeMap<String, Value>, String>> = LazyLock::new(parse_item_schemas);

/// Compiled input and output item validators.
static ITEM_VALIDATORS: LazyLock<Result<ItemValidators, String>> = LazyLock::new(compile_item_validators);

/// Generated artifact wire shape.
#[derive(Deserialize)]
struct ItemContractArtifact {
    /// Artifact schema version.
    schema_version: u8,

    /// Recursive `OpenAPI` component closure.
    schemas: BTreeMap<String, Value>,
}

/// Validators for request and response item boundaries.
struct ItemValidators {
    /// Official `InputItem` validator.
    input: Validator,

    /// Official `ConversationItem` validator.
    output: Validator,
}

/// Return generated schemas for insertion into the implementation document.
pub(super) fn openapi_components() -> Result<Map<String, Value>, String> {
    ITEM_SCHEMAS
        .as_ref()
        .map(|schemas| {
            schemas
                .iter()
                .map(|(name, schema)| (name.clone(), schema.clone()))
                .collect()
        })
        .map_err(Clone::clone)
}

/// Validate one item accepted by a create operation.
pub(super) fn validate_input_item(item: &Value) -> Result<(), String> {
    let validators = ITEM_VALIDATORS.as_ref().map_err(Clone::clone)?;
    validate_item(&validators.input, item, "InputItem")
}

/// Validate one normalized item returned by the local API.
pub(super) fn validate_output_item(item: &Value) -> Result<(), String> {
    let validators = ITEM_VALIDATORS.as_ref().map_err(Clone::clone)?;
    validate_item(&validators.output, item, "ConversationItem")
}

/// Compile both validators from the same generated component closure.
fn compile_item_validators() -> Result<ItemValidators, String> {
    let schemas = ITEM_SCHEMAS.as_ref().map_err(Clone::clone)?;
    Ok(ItemValidators {
        input: compile_validator(schemas, "InputItem")?,
        output: compile_validator(schemas, "ConversationItem")?,
    })
}

/// Compile one root reference against the generated components.
fn compile_validator(schemas: &BTreeMap<String, Value>, root: &str) -> Result<Validator, String> {
    let mut schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": format!("#/components/schemas/{root}"),
        "components": {"schemas": schemas},
    });
    // EasyInputMessage and Item overlap for list-form messages, so runtime
    // accepts membership in either official branch while OpenAPI keeps oneOf.
    if root == "InputItem" {
        let input = schema
            .pointer_mut("/components/schemas/InputItem")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "generated item contracts have no InputItem object".to_owned())?;
        let variants = input
            .remove("oneOf")
            .ok_or_else(|| "generated InputItem contract has no oneOf variants".to_owned())?;
        input.insert("anyOf".to_owned(), variants);
    }
    jsonschema::options()
        .with_draft(Draft::Draft202012)
        .build(&schema)
        .map_err(|error| format!("failed to compile {root} schema: {error}"))
}

/// Parse and version-check the generated artifact.
fn parse_item_schemas() -> Result<BTreeMap<String, Value>, String> {
    let artifact: ItemContractArtifact = serde_json::from_str(ITEM_CONTRACTS_JSON)
        .map_err(|error| format!("failed to parse generated item contracts: {error}"))?;
    if artifact.schema_version != 1 {
        return Err(format!(
            "unsupported Conversation item contract schema version {}",
            artifact.schema_version
        ));
    }
    Ok(artifact.schemas)
}

/// Render the first schema error as an invalid-request diagnostic.
fn validate_item(validator: &Validator, item: &Value, contract: &str) -> Result<(), String> {
    validator
        .validate(item)
        .map_err(|error| format!("item does not match {contract}: {error}"))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn generated_contracts_accept_representative_items() {
        validate_input_item(&json!({"type": "message", "role": "user", "content": "hello"})).unwrap();
        validate_input_item(&json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}"
        }))
        .unwrap();
        validate_input_item(&json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "done"
        }))
        .unwrap();
        validate_input_item(&json!({
            "type": "program",
            "id": "prog_1",
            "call_id": "call_2",
            "code": "return 1",
            "fingerprint": "fp_1"
        }))
        .unwrap();
        validate_output_item(&json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": []
        }))
        .unwrap();
    }

    #[test]
    fn generated_contracts_reject_unknown_and_malformed_items() {
        let unknown = validate_input_item(&json!({"type": "future_item"})).unwrap_err();
        assert!(
            unknown.contains("InputItem"),
            "unknown type should fail InputItem validation: {unknown}"
        );

        let malformed = validate_input_item(&json!({"type": "function_call"})).unwrap_err();
        assert!(
            malformed.contains("InputItem"),
            "missing required fields should fail InputItem validation: {malformed}"
        );
    }
}
