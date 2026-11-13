// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Generated `OpenAPI` description for the Praxis-owned Files contract.

use utoipa::openapi::OpenApi;

use super::routes::operation_specs;
use crate::openai::operation;

/// Generate the Files implementation `OpenAPI` document as pretty JSON.
///
/// # Errors
///
/// Returns an error if generated schemas cannot be assembled or serialized.
pub fn implementation_openapi_json() -> Result<String, String> {
    serde_json::to_string_pretty(&implementation_openapi())
        .map_err(|error| format!("failed to serialize Files OpenAPI: {error}"))
}

/// Build the Files implementation document from the operation registry and
/// its bound runtime contract types.
fn implementation_openapi() -> OpenApi {
    operation::implementation_openapi(
        "Praxis AI OpenAI Files implementation",
        "0.1.0",
        "Files",
        operation_specs().iter().map(|spec| &spec.definition),
    )
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::Value;

    use super::*;
    use crate::openai::operation::collect_component_references;

    /// Generated document as JSON.
    fn document() -> Value {
        serde_json::from_str(&implementation_openapi_json().unwrap()).unwrap()
    }

    /// Strings in a JSON array.
    fn strings(value: &Value) -> BTreeSet<&str> {
        value.as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect()
    }

    #[test]
    fn every_component_reference_resolves() {
        let document = document();
        let mut references = Vec::new();
        collect_component_references(&document, &mut references);
        assert!(!references.is_empty());
        for reference in references {
            assert!(
                document.pointer(reference.strip_prefix('#').unwrap()).is_some(),
                "unresolved reference: {reference}"
            );
        }
    }

    #[test]
    fn document_shape() {
        let document = document();
        assert_eq!(document["openapi"], "3.1.0");
        assert_eq!(document["tags"].as_array().unwrap().len(), 1);
        assert_eq!(document["tags"][0]["name"], "Files");
        for (path, item) in document["paths"].as_object().unwrap() {
            for (method, operation) in item.as_object().unwrap() {
                let statuses = operation["responses"].as_object().unwrap().keys().collect::<Vec<_>>();
                assert!(statuses.contains(&&"200".to_owned()), "{method} {path}");
            }
        }
    }

    #[test]
    fn create_file_operation() {
        let document = document();
        let operation = &document["paths"]["/files"]["post"];
        assert!(operation.get("parameters").is_none());
        assert_eq!(operation["requestBody"]["required"], true);
        let content = operation["requestBody"]["content"].as_object().unwrap();
        assert_eq!(content.keys().collect::<Vec<_>>(), ["multipart/form-data"]);
        assert_eq!(
            content["multipart/form-data"]["schema"]["$ref"],
            "#/components/schemas/CreateFileRequest"
        );
        assert_eq!(
            operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/OpenAIFile"
        );
    }

    #[test]
    fn list_files_operation() {
        let document = document();
        let operation = &document["paths"]["/files"]["get"];
        let parameters = operation["parameters"].as_array().unwrap();
        let names = parameters
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, ["purpose", "limit", "order", "after"]);
        for parameter in parameters {
            assert_eq!(parameter["in"], "query");
            assert_eq!(parameter["required"], false);
        }
        let limit = &parameters[1]["schema"];
        assert_eq!(limit["type"], "integer");
        assert!(limit.get("minimum").is_none() && limit.get("maximum").is_none());
        assert_eq!(limit["default"], 10_000);
        let order = &parameters[2]["schema"];
        assert_eq!(strings(&order["enum"]), BTreeSet::from(["asc", "desc"]));
        assert_eq!(order["default"], "desc");
        assert_eq!(
            operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/ListFilesResponse"
        );
    }

    #[test]
    fn file_id_operations() {
        let document = document();
        for (path, method, content_type, schema) in [
            ("/files/{file_id}", "get", "application/json", "OpenAIFile"),
            ("/files/{file_id}", "delete", "application/json", "DeleteFileResponse"),
            (
                "/files/{file_id}/content",
                "get",
                "application/octet-stream",
                "FileContent",
            ),
        ] {
            let operation = &document["paths"][path][method];
            let parameters = operation["parameters"].as_array().unwrap();
            assert_eq!(parameters.len(), 1, "{method} {path}");
            assert_eq!(parameters[0]["name"], "file_id");
            assert_eq!(parameters[0]["in"], "path");
            assert_eq!(parameters[0]["required"], true);
            assert_eq!(parameters[0]["schema"]["type"], "string");
            let content = operation["responses"]["200"]["content"].as_object().unwrap();
            assert_eq!(content.keys().collect::<Vec<_>>(), [content_type], "{method} {path}");
            assert_eq!(
                content[content_type]["schema"]["$ref"],
                format!("#/components/schemas/{schema}")
            );
        }
    }

    #[test]
    fn create_file_request_component() {
        let document = document();
        let schema = &document["components"]["schemas"]["CreateFileRequest"];
        assert_eq!(strings(&schema["required"]), BTreeSet::from(["file", "purpose"]));
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["file"]["type"], "string");
        assert_eq!(schema["properties"]["file"]["format"], "binary");
        let purpose = &document["components"]["schemas"]["UploadPurpose"];
        assert_eq!(
            strings(&purpose["enum"]),
            BTreeSet::from(["assistants", "batch", "fine-tune", "vision", "user_data", "evals"])
        );
        let expires = &schema["properties"]["expires_after"];
        assert_eq!(expires["$ref"], "#/components/schemas/FileExpirationAfter");
        assert!(expires.get("oneOf").is_none() && expires.get("type").is_none());
    }

    #[test]
    fn expiration_component() {
        let document = document();
        let schema = &document["components"]["schemas"]["FileExpirationAfter"];
        assert_eq!(strings(&schema["required"]), BTreeSet::from(["anchor", "seconds"]));
        let anchor = &document["components"]["schemas"]["ExpirationAnchor"];
        assert_eq!(strings(&anchor["enum"]), BTreeSet::from(["created_at"]));
        assert_eq!(schema["properties"]["seconds"]["minimum"], 3600);
        assert_eq!(schema["properties"]["seconds"]["maximum"], 2_592_000);
    }

    #[test]
    fn file_component() {
        let document = document();
        let schemas = &document["components"]["schemas"];
        let schema = &schemas["OpenAIFile"];
        assert_eq!(
            strings(&schema["required"]),
            BTreeSet::from(["id", "object", "bytes", "created_at", "filename", "purpose", "status"])
        );
        let properties = &schema["properties"];
        assert_eq!(strings(&properties["object"]["enum"]), BTreeSet::from(["file"]));
        assert_eq!(properties["created_at"]["format"], "unixtime");
        assert_eq!(strings(&schemas["FilePurpose"]["enum"]).len(), 9);
        assert!(strings(&schemas["FilePurpose"]["enum"]).contains("evals"));
        assert_eq!(properties["status"]["deprecated"], true);
        assert_eq!(properties["status_details"]["deprecated"], true);
        assert_eq!(properties["status_details"]["type"], "string");
        assert_eq!(properties["expires_at"]["type"], "integer");
    }

    #[test]
    fn list_response_component() {
        let document = document();
        let schema = &document["components"]["schemas"]["ListFilesResponse"];
        assert_eq!(
            strings(&schema["required"]),
            BTreeSet::from(["object", "data", "first_id", "last_id", "has_more"])
        );
        let properties = &schema["properties"];
        assert_eq!(properties["object"]["type"], "string");
        assert!(properties["object"].get("enum").is_none());
        assert_eq!(properties["first_id"]["type"], "string");
        assert_eq!(properties["last_id"]["type"], "string");
        assert_eq!(properties["data"]["items"]["$ref"], "#/components/schemas/OpenAIFile");
    }

    #[test]
    fn delete_response_and_content_components() {
        let document = document();
        let schemas = &document["components"]["schemas"];
        assert_eq!(
            strings(&schemas["DeleteFileResponse"]["required"]),
            BTreeSet::from(["id", "object", "deleted"])
        );
        assert_eq!(
            strings(&schemas["DeleteFileResponse"]["properties"]["object"]["enum"]),
            BTreeSet::from(["file"])
        );
        assert_eq!(schemas["FileContent"]["type"], "string");
        assert_eq!(schemas["FileContent"]["format"], "binary");
    }
}
