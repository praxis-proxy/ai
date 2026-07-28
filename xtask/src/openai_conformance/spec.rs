// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};

use super::{
    HTTP_METHODS,
    model::{OperationKey, OperationScope, ReferenceProvenance, SpecOperation, SpecSourceReport},
    reference::{ReferenceDocument, load_reference_provenance, sha256_hex},
};

/// Complete checked-in OpenAI source shared by every selected area.
pub(super) struct ReferenceSource {
    /// Stable source path.
    pub(super) source: String,

    /// Parsed complete `OpenAPI` document shared by area projections.
    pub(super) document: ReferenceDocument,

    /// Immutable upstream identity, absent for a CLI override.
    pub(super) provenance: Option<ReferenceProvenance>,

    /// Digest of the complete source document.
    pub(super) source_sha256: String,
}

/// One semantic area projection and its parsed operations.
pub(super) struct AreaReference {
    /// Report metadata for the projection.
    pub(super) report: SpecSourceReport,

    /// Operations selected into the projection.
    pub(super) operations: Vec<SpecOperation>,

    /// Deterministic self-contained `OpenAPI` projection.
    pub(super) content: String,
}

/// Load the complete spec source and verify its optional provenance manifest.
pub(super) fn load_reference_source(source: &str, manifest_source: Option<&str>) -> Result<ReferenceSource, String> {
    let content = read_spec(source)?;
    let (provenance, source_sha256) = if let Some(manifest_source) = manifest_source {
        let (provenance, digest) = load_reference_provenance(manifest_source, &content)?;
        (Some(provenance), digest)
    } else {
        (None, sha256_hex(content.as_bytes()))
    };
    let document = ReferenceDocument::parse(content.as_bytes())?;
    Ok(ReferenceSource {
        source: source.to_owned(),
        document,
        provenance,
        source_sha256,
    })
}

/// Build and parse one deterministic area projection.
pub(super) fn project_reference(reference: &ReferenceSource, scope: OperationScope) -> Result<AreaReference, String> {
    let content = reference.document.project(scope)?;
    let operations = scope_operations(parse_openapi_operations(&content)?, scope);

    if operations.is_empty() {
        return Err(format!(
            "{} spec {source} did not contain any {} operations",
            scope.label,
            scope.label,
            source = reference.source,
        ));
    }

    let report = SpecSourceReport {
        area_id: scope.id,
        area: scope.label,
        operations: operations.len(),
        projection_sha256: sha256_hex(content.as_bytes()),
    };
    Ok(AreaReference {
        report,
        operations,
        content,
    })
}

/// Keep only operations for the requested scope and tag them with the area.
pub(super) fn scope_operations(operations: Vec<SpecOperation>, scope: OperationScope) -> Vec<SpecOperation> {
    let mut scoped: Vec<SpecOperation> = operations
        .into_iter()
        .filter(|operation| scope.matches(&operation.key.path))
        .map(|mut operation| {
            operation.area = scope.label;
            operation
        })
        .collect();
    scoped.sort_by(|a, b| a.key.cmp(&b.key));
    scoped
}

/// Read the `OpenAPI` document from a local path.
pub(super) fn read_spec(source: &str) -> Result<String, String> {
    if is_url(source) {
        return Err(format!(
            "remote OpenAPI spec sources are not supported ({source}); vendor the spec under docs/conformance/specs/ and pass a local path"
        ));
    }

    let path = spec_path(source);
    std::fs::read_to_string(&path).map_err(|e| format!("failed to read {}: {e}", path.display()))
}

/// Resolve a spec path relative to the repository root.
fn spec_path(source: &str) -> PathBuf {
    let path = Path::new(source);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    repo_root().join(path)
}

/// Repository root inferred from the `xtask` manifest location.
pub(super) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest should have a repository parent")
        .to_path_buf()
}

/// Return whether a spec source is an HTTP URL.
fn is_url(source: &str) -> bool {
    source.starts_with("https://") || source.starts_with("http://")
}

/// Parse `OpenAPI` operations from a YAML or JSON document.
pub(super) fn parse_openapi_operations(content: &str) -> Result<Vec<SpecOperation>, String> {
    if content.trim_start().starts_with('{') {
        return parse_json_openapi_operations(content);
    }

    parse_yaml_openapi_operations(content)
}

/// Parse `OpenAPI` operations from a JSON document.
fn parse_json_openapi_operations(content: &str) -> Result<Vec<SpecOperation>, String> {
    let spec: Value = serde_json::from_str(content).map_err(|e| format!("failed to parse OpenAPI JSON: {e}"))?;
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenAPI document does not contain a paths object".to_owned())?;

    let mut operations = Vec::new();
    for (path, path_item) in paths {
        let Some(path_item) = path_item.as_object() else {
            continue;
        };
        for method in HTTP_METHODS {
            let Some(operation) = path_item.get(*method).and_then(Value::as_object) else {
                continue;
            };
            operations.push(parse_json_operation(path, method, operation));
        }
    }
    operations.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(operations)
}

/// Parse `OpenAPI` operations from a YAML document.
fn parse_yaml_openapi_operations(content: &str) -> Result<Vec<SpecOperation>, String> {
    let spec: YamlValue = serde_yaml::from_str(content).map_err(|e| format!("failed to parse OpenAPI YAML: {e}"))?;
    let paths = yaml_mapping_get(
        spec.as_mapping()
            .ok_or_else(|| "OpenAPI YAML document is not an object".to_owned())?,
        "paths",
    )
    .and_then(YamlValue::as_mapping)
    .ok_or_else(|| "OpenAPI document does not contain a paths object".to_owned())?;

    let mut operations = Vec::new();
    for (path, path_item) in paths {
        let Some(path) = path.as_str() else {
            continue;
        };
        let Some(path_item) = path_item.as_mapping() else {
            continue;
        };
        for method in HTTP_METHODS {
            let Some(operation) = yaml_mapping_get(path_item, method).and_then(YamlValue::as_mapping) else {
                continue;
            };
            operations.push(parse_yaml_operation(path, method, operation));
        }
    }
    operations.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(operations)
}

/// Parse a single JSON `OpenAPI` operation object.
fn parse_json_operation(path: &str, method: &str, operation: &Map<String, Value>) -> SpecOperation {
    build_operation(
        path,
        method,
        first_json_tag(operation),
        operation.get("operationId").and_then(Value::as_str).map(str::to_owned),
        operation.get("deprecated").and_then(Value::as_bool).unwrap_or(false),
    )
}

/// Parse a single YAML `OpenAPI` operation object.
fn parse_yaml_operation(path: &str, method: &str, operation: &YamlMapping) -> SpecOperation {
    build_operation(
        path,
        method,
        first_yaml_tag(operation),
        yaml_mapping_get(operation, "operationId")
            .and_then(YamlValue::as_str)
            .map(str::to_owned),
        yaml_mapping_get(operation, "deprecated")
            .and_then(YamlValue::as_bool)
            .unwrap_or(false),
    )
}

/// Build one operation from format-agnostic fields.
fn build_operation(
    path: &str,
    method: &str,
    tag: String,
    operation_id: Option<String>,
    deprecated: bool,
) -> SpecOperation {
    SpecOperation {
        key: OperationKey::new(method.to_ascii_uppercase(), path),
        tag,
        area: "unscoped",
        operation_id,
        deprecated,
        beta: path.contains("?beta=true"),
    }
}

/// Return the JSON operation's first tag, or `untagged`.
fn first_json_tag(operation: &Map<String, Value>) -> String {
    operation
        .get("tags")
        .and_then(Value::as_array)
        .and_then(|tags| tags.first())
        .and_then(Value::as_str)
        .unwrap_or("untagged")
        .to_owned()
}

/// Return the YAML operation's first tag, or `untagged`.
fn first_yaml_tag(operation: &YamlMapping) -> String {
    yaml_mapping_get(operation, "tags")
        .and_then(YamlValue::as_sequence)
        .and_then(|tags| tags.first())
        .and_then(YamlValue::as_str)
        .unwrap_or("untagged")
        .to_owned()
}

/// Get a string-keyed value from a YAML mapping.
fn yaml_mapping_get<'a>(mapping: &'a YamlMapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_owned()))
}
