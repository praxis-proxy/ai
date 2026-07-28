// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Generated `OpenAPI` description for locally owned Conversations operations.

use utoipa::openapi::OpenApi;

use super::routes::operation_specs;
use crate::openai::operation;

/// Generate the local Conversations implementation `OpenAPI` document as
/// pretty JSON.
///
/// # Errors
///
/// Returns an error if the generated `OpenAPI` document cannot be serialized
/// as JSON.
pub fn implementation_openapi_json() -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&implementation_openapi())
}

/// Build the local Conversations implementation document from the operation
/// registry and its bound runtime contract types.
fn implementation_openapi() -> OpenApi {
    operation::implementation_openapi(
        "Praxis AI OpenAI Conversations implementation",
        "0.1.0",
        "Conversations",
        operation_specs().iter().map(|spec| &spec.definition),
    )
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::Value;

    use super::*;

    #[test]
    fn generated_operations_come_from_owned_registry_entries() {
        let openapi = implementation_openapi();
        let generated = openapi
            .paths
            .paths
            .iter()
            .flat_map(|(path, item)| {
                [
                    ("GET", item.get.as_ref()),
                    ("POST", item.post.as_ref()),
                    ("DELETE", item.delete.as_ref()),
                ]
                .into_iter()
                .filter_map(move |(method, operation)| operation.map(|_| (method, path.as_str())))
            })
            .collect::<BTreeSet<_>>();
        let expected = operation_specs()
            .iter()
            .filter(|spec| spec.owned_contract().is_some() && spec.mode.owns_contract())
            .map(|spec| (spec.method.as_str(), spec.spec_path))
            .collect::<BTreeSet<_>>();

        assert_eq!(generated, expected);
    }

    #[test]
    fn every_generated_component_reference_resolves() {
        let document = serde_json::to_value(implementation_openapi()).unwrap();
        let mut references = Vec::new();
        collect_component_references(&document, &mut references);
        assert!(!references.is_empty());

        for reference in references {
            let pointer = reference.strip_prefix('#').unwrap();
            assert!(
                document.pointer(pointer).is_some(),
                "generated OpenAPI reference does not resolve: {reference}"
            );
        }
    }

    #[test]
    fn generated_components_preserve_runtime_contract_constraints() {
        let document = serde_json::to_value(implementation_openapi()).unwrap();

        assert_eq!(
            document.pointer("/components/schemas/CreateConversationRequest/properties/items/type"),
            Some(&Value::String("array".to_owned()))
        );
        assert_eq!(
            document.pointer("/components/schemas/CreateConversationItemsRequest/properties/items/type"),
            Some(&Value::String("array".to_owned()))
        );
        assert_eq!(
            document.pointer("/components/schemas/ConversationItem/type"),
            Some(&Value::String("object".to_owned()))
        );
        assert_eq!(
            document.pointer("/components/schemas/ConversationResource/properties/created_at/format"),
            Some(&Value::String("unixtime".to_owned()))
        );
        assert!(
            document
                .pointer("/components/schemas/ConversationItemList/properties/object/default")
                .is_none(),
            "the list discriminator has no runtime defaulting behavior"
        );
        assert!(
            document.pointer("/paths/~1conversations/post/parameters").is_none(),
            "parameterless operations should omit the OpenAPI parameters field"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "checks all three operations and eight enum values")]
    fn generated_item_operations_share_the_official_include_enum() {
        let document = serde_json::to_value(implementation_openapi()).unwrap();
        let expected = [
            "file_search_call.results",
            "web_search_call.results",
            "web_search_call.action.sources",
            "message.input_image.image_url",
            "computer_call_output.output.image_url",
            "code_interpreter_call.outputs",
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        ]
        .map(Value::from)
        .to_vec();
        for pointer in [
            "/paths/~1conversations~1{conversation_id}~1items/post/parameters",
            "/paths/~1conversations~1{conversation_id}~1items/get/parameters",
            "/paths/~1conversations~1{conversation_id}~1items~1{item_id}/get/parameters",
        ] {
            let parameters = document.pointer(pointer).and_then(Value::as_array).unwrap();
            let include = parameters
                .iter()
                .find(|parameter| parameter.get("name") == Some(&Value::String("include".to_owned())))
                .unwrap();
            assert_eq!(include["in"], "query");
            assert_eq!(include["required"], false);
            assert_eq!(
                include.pointer("/schema/type"),
                Some(&Value::String("array".to_owned()))
            );
            assert_eq!(
                include.pointer("/schema/items/enum"),
                Some(&Value::Array(expected.clone()))
            );
        }
    }

    fn collect_component_references<'a>(value: &'a Value, references: &mut Vec<&'a str>) {
        match value {
            Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(Value::as_str)
                    && reference.starts_with("#/components/schemas/")
                {
                    references.push(reference);
                }
                for value in object.values() {
                    collect_component_references(value, references);
                }
            },
            Value::Array(values) => {
                for value in values {
                    collect_component_references(value, references);
                }
            },
            _ => {},
        }
    }
}
