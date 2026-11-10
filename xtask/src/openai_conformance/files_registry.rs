// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Three-way drift check for the Files operations.
//!
//! The registry is the runtime source of truth for Files operation identity.
//! This check compares `(method, path, operationId)` across three sources and
//! fails when any of them disagree:
//!
//! - **pinned**: the Files operations in the pinned OpenAI specification;
//! - **runtime**: the Files operation registry;
//! - **generated**: the `OpenAPI` document generated from that registry.
//!
//! Parameters, bodies, and response schemas are compared separately by
//! `cargo xtask openai-conformance` (oasdiff).

#[cfg(feature = "openai-all")]
use super::HTTP_METHODS;
use super::{
    area::{FILES_SCOPE, OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC},
    registry_check::{OperationIds, compare_three_way},
    spec::load_reference_source,
};

/// Compare pinned, runtime, and generated Files operations.
pub(super) fn check() -> Result<String, String> {
    let reference = load_reference_source(OPENAI_REFERENCE_SPEC, Some(OPENAI_REFERENCE_MANIFEST))?;
    let mut pinned = OperationIds::new();
    let mut missing_ids = Vec::new();
    for operation in reference.document.scoped_operations(FILES_SCOPE)? {
        let key = (operation.key.method, operation.key.path);
        match operation.operation_id {
            Some(id) => {
                pinned.insert(key, id);
            },
            None => missing_ids.push(format!("pinned operation {} {} has no operationId", key.0, key.1)),
        }
    }
    let runtime: OperationIds = praxis_ai_apis::openai::files_operation_specs()
        .iter()
        .map(|spec| {
            (
                (spec.method().as_str().to_owned(), spec.spec_path.to_owned()),
                spec.operation_id().to_owned(),
            )
        })
        .collect();
    let generated = generated_operations()?;

    let mut failures = missing_ids;
    failures.extend(compare_three_way(&pinned, &runtime, &generated));
    if failures.is_empty() {
        Ok(format!(
            "files registry, generated document, and pinned specification agree: {} operations checked",
            runtime.len()
        ))
    } else {
        Err(failures.join("\n"))
    }
}

/// Operation identities in the generated Files document.
fn generated_operations() -> Result<OperationIds, String> {
    #[cfg(feature = "openai-all")]
    {
        let json = praxis_ai_apis::openai::files_openapi_json()
            .map_err(|e| format!("failed to generate Files implementation OpenAPI spec: {e}"))?;
        parse_operations(&json)
    }
    #[cfg(not(feature = "openai-all"))]
    {
        Err(
            "Files registry check requires the experimental APIs: use cargo xtask or cargo run -p xtask --features openai-all --"
                .to_owned(),
        )
    }
}

/// Extract `(METHOD, path) -> operationId` from a generated `OpenAPI` document.
#[cfg(feature = "openai-all")]
fn parse_operations(json: &str) -> Result<OperationIds, String> {
    let document: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("generated Files OpenAPI is not valid JSON: {e}"))?;
    let paths = document
        .get("paths")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "generated Files OpenAPI has no paths object".to_owned())?;
    let mut operations = OperationIds::new();
    for (path, item) in paths {
        for method in HTTP_METHODS {
            let Some(operation) = item.get(*method) else { continue };
            let id = operation
                .get("operationId")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("generated {method} {path} has no operationId"))?;
            operations.insert((method.to_uppercase(), path.clone()), id.to_owned());
        }
    }
    Ok(operations)
}
