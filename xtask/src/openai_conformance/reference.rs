// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Verification and semantic projection of the pinned complete OpenAI spec.

use std::{collections::BTreeSet, path::Path};

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{
    HTTP_METHODS,
    area::{OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC},
    model::{OperationScope, ReferenceProvenance},
    semantic_yaml::{Mapping as YamlMapping, Value as YamlValue},
    spec::repo_root,
};

/// Pinned upstream repository used for the complete OpenAI reference.
const OPENAI_OPENAPI_REPOSITORY: &str = "https://github.com/openai/openai-openapi";
/// Upstream document path.
const OPENAI_OPENAPI_PATH: &str = "openapi.yaml";

/// CLI arguments for `cargo xtask openai-conformance-reference`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Pin a new 40-character upstream commit and vendor the complete spec.
    #[arg(long, value_name = "COMMIT", conflicts_with = "check")]
    revision: Option<String>,

    /// Verify the checked-in complete spec against its immutable upstream pin.
    #[arg(long)]
    check: bool,
}

/// Checked-in source pin for the complete OpenAI reference.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReferenceManifest {
    /// Manifest schema version.
    schema_version: u32,
    /// Upstream repository.
    repository: String,
    /// Pinned upstream commit.
    revision: String,
    /// Path within the upstream repository.
    path: String,
    /// SHA-256 of the complete upstream document.
    source_sha256: String,
}

impl ReferenceManifest {
    /// Convert the manifest into report provenance.
    fn provenance(&self) -> ReferenceProvenance {
        ReferenceProvenance {
            repository: self.repository.clone(),
            revision: self.revision.clone(),
            path: self.path.clone(),
            source_sha256: self.source_sha256.clone(),
        }
    }
}

/// Run the reference source task.
pub(crate) fn run(args: &Args) {
    let result = if args.check {
        check_reference_source()
    } else if let Some(revision) = &args.revision {
        refresh_reference_source(revision)
    } else {
        Err("pass either --revision <40-character-commit> or --check".to_owned())
    };

    result.unwrap_or_else(|e| {
        eprintln!("openai-conformance-reference failed: {e}");
        std::process::exit(1);
    });
}

/// Load checked-in provenance and verify the complete source digest.
pub(super) fn load_reference_provenance(
    manifest_source: &str,
    source_content: &str,
) -> Result<(ReferenceProvenance, String), String> {
    let manifest = read_manifest(&repo_root().join(manifest_source))?;
    validate_manifest(&manifest)?;
    let digest = sha256_hex(source_content.as_bytes());
    if digest != manifest.source_sha256 {
        return Err(format!(
            "checked-in OpenAI source digest mismatch: expected {}, got {digest}",
            manifest.source_sha256,
        ));
    }
    Ok((manifest.provenance(), digest))
}

/// Vendor the complete OpenAI spec from a new immutable upstream revision.
fn refresh_reference_source(revision: &str) -> Result<(), String> {
    validate_revision(revision)?;
    let source = fetch_upstream(revision)?;
    let manifest = ReferenceManifest {
        schema_version: 1,
        repository: OPENAI_OPENAPI_REPOSITORY.to_owned(),
        revision: revision.to_owned(),
        path: OPENAI_OPENAPI_PATH.to_owned(),
        source_sha256: sha256_hex(&source),
    };
    write_artifacts(&manifest, &source)?;
    println!("updated {OPENAI_REFERENCE_SPEC} from {revision}");
    println!("updated {OPENAI_REFERENCE_MANIFEST}");
    Ok(())
}

/// Verify the checked-in complete spec against its pinned manifest digest.
fn check_reference_source() -> Result<(), String> {
    let manifest_path = repo_root().join(OPENAI_REFERENCE_MANIFEST);
    let manifest = read_manifest(&manifest_path)?;
    validate_manifest(&manifest)?;

    let source_path = repo_root().join(OPENAI_REFERENCE_SPEC);
    let checked_in =
        std::fs::read(&source_path).map_err(|e| format!("failed to read {}: {e}", source_path.display()))?;
    let checked_in_sha256 = sha256_hex(&checked_in);
    if checked_in_sha256 != manifest.source_sha256 {
        return Err(format!(
            "{} digest mismatch: expected {}, got {checked_in_sha256}; \
             run cargo xtask openai-conformance-reference --revision {}",
            source_path.display(),
            manifest.source_sha256,
            manifest.revision,
        ));
    }

    println!(
        "complete OpenAI reference matches pinned revision {}",
        manifest.revision
    );
    Ok(())
}

/// Fetch the one permitted upstream document at an immutable revision.
fn fetch_upstream(revision: &str) -> Result<Vec<u8>, String> {
    validate_revision(revision)?;
    let url = format!("https://raw.githubusercontent.com/openai/openai-openapi/{revision}/{OPENAI_OPENAPI_PATH}");
    let response = reqwest::blocking::get(&url).map_err(|e| format!("failed to fetch {url}: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("failed to fetch {url}: HTTP {status}"));
    }
    response
        .bytes()
        .map(|bytes| bytes.to_vec())
        .map_err(|e| format!("failed to read {url}: {e}"))
}

/// Parsed complete OpenAI document reused by every selected area.
pub(super) struct ReferenceDocument {
    /// Semantic YAML tree retaining `OpenAPI` numeric bounds outside `i64`.
    document: YamlValue,
}

impl ReferenceDocument {
    /// Parse and validate the complete upstream `OpenAPI` document once.
    pub(super) fn parse(source: &[u8]) -> Result<Self, String> {
        let source = std::str::from_utf8(source).map_err(|e| format!("upstream OpenAPI is not UTF-8: {e}"))?;
        let document: YamlValue =
            serde_yaml::from_str(source).map_err(|e| format!("failed to parse complete upstream OpenAPI: {e}"))?;
        if document.as_mapping().is_none() {
            return Err("upstream OpenAPI document is not an object".to_owned());
        }
        Ok(Self { document })
    }

    /// Build a self-contained, stable area projection.
    #[expect(clippy::too_many_lines, reason = "ordered projection assembly")]
    pub(super) fn project(&self, scope: OperationScope) -> Result<String, String> {
        let root = self
            .document
            .as_mapping()
            .expect("validated by ReferenceDocument::parse");

        let mut bundle = YamlMapping::new();
        copy_root_field(root, &mut bundle, "openapi");
        copy_root_field(root, &mut bundle, "jsonSchemaDialect");
        copy_root_field(root, &mut bundle, "info");
        copy_root_field(root, &mut bundle, "security");

        let source_paths = root
            .get(&YamlValue::String("paths".to_owned()))
            .and_then(YamlValue::as_mapping)
            .ok_or_else(|| "upstream OpenAPI document does not contain a paths object".to_owned())?;
        let selected_paths = source_paths
            .iter()
            .filter(|(path, _)| path.as_str().is_some_and(|path| scope.matches(path)))
            .map(|(path, item)| (path.clone(), item.clone()))
            .collect::<YamlMapping>();
        if selected_paths.is_empty() {
            return Err(format!("upstream OpenAPI did not contain any {} paths", scope.label));
        }
        bundle.insert(
            YamlValue::String("paths".to_owned()),
            YamlValue::Mapping(selected_paths),
        );

        let mut refs = BTreeSet::new();
        collect_refs(yaml_get(&bundle, "paths").expect("paths inserted above"), &mut refs)?;
        if let Some(security) = yaml_get(&bundle, "security") {
            collect_security_scheme_refs(security, &mut refs);
        }
        collect_operation_security_scheme_refs(yaml_get(&bundle, "paths").expect("paths inserted above"), &mut refs);

        let components = collect_components(root, refs)?;
        if !components.is_empty() {
            bundle.insert(
                YamlValue::String("components".to_owned()),
                YamlValue::Mapping(components),
            );
        }

        let content = serde_yaml::to_string(&YamlValue::Mapping(bundle))
            .map_err(|e| format!("failed to serialize reference projection: {e}"))?;
        Ok(ensure_trailing_newline(content))
    }
}

/// Parse and project one complete source in a single call.
#[cfg(test)]
pub(super) fn build_reference_projection(source: &[u8], scope: OperationScope) -> Result<String, String> {
    ReferenceDocument::parse(source)?.project(scope)
}

/// Recursively close over local component references.
fn collect_components(root: &YamlMapping, mut pending: BTreeSet<String>) -> Result<YamlMapping, String> {
    let mut output = YamlMapping::new();
    let mut seen = BTreeSet::new();
    let source_components = yaml_get(root, "components")
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| "upstream OpenAPI document does not contain components".to_owned())?;

    while let Some(reference) = pending.pop_first() {
        let (kind, name) = component_ref_parts(&reference)?;
        let component_key = format!("{kind}/{name}");
        if !seen.insert(component_key) {
            continue;
        }
        let value = yaml_get(source_components, &kind)
            .and_then(YamlValue::as_mapping)
            .and_then(|values| yaml_get(values, &name))
            .cloned()
            .ok_or_else(|| format!("failed to resolve {reference}: component {kind}/{name} was not found"))?;
        collect_refs(&value, &mut pending)?;

        let kind_key = YamlValue::String(kind);
        if !output.contains_key(&kind_key) {
            output.insert(kind_key.clone(), YamlValue::Mapping(YamlMapping::new()));
        }
        output
            .get_mut(&kind_key)
            .and_then(YamlValue::as_mapping_mut)
            .expect("component category inserted as object")
            .insert(YamlValue::String(name), value);
    }

    Ok(output)
}

/// Collect all `$ref` strings and reject anything that could access a resource.
fn collect_refs(value: &YamlValue, refs: &mut BTreeSet<String>) -> Result<(), String> {
    if let Some(fields) = value.as_mapping() {
        if let Some(reference) = yaml_get(fields, "$ref").and_then(YamlValue::as_str) {
            if !reference.starts_with("#/components/") {
                return Err(format!("unsupported external or non-component reference {reference}"));
            }
            refs.insert(reference.to_owned());
        }
        for child in fields.values() {
            collect_refs(child, refs)?;
        }
    } else if let Some(values) = value.as_sequence() {
        for child in values {
            collect_refs(child, refs)?;
        }
    }
    Ok(())
}

/// Add security schemes named by one `OpenAPI` security requirement array.
fn collect_security_scheme_refs(security: &YamlValue, refs: &mut BTreeSet<String>) {
    let Some(requirements) = security.as_sequence() else {
        return;
    };
    for requirement in requirements.iter().filter_map(YamlValue::as_mapping) {
        for name in requirement.keys() {
            if let Some(name) = name.as_str() {
                refs.insert(format!("#/components/securitySchemes/{}", escape_json_pointer(name)));
            }
        }
    }
}

/// Add schemes named by operation-level security overrides.
fn collect_operation_security_scheme_refs(paths: &YamlValue, refs: &mut BTreeSet<String>) {
    let Some(paths) = paths.as_mapping() else {
        return;
    };
    for path_item in paths.values().filter_map(YamlValue::as_mapping) {
        for method in HTTP_METHODS {
            if let Some(security) = yaml_get(path_item, method)
                .and_then(YamlValue::as_mapping)
                .and_then(|operation| yaml_get(operation, "security"))
            {
                collect_security_scheme_refs(security, refs);
            }
        }
    }
}

/// Split `#/components/<kind>/<name>` and decode JSON pointer escapes.
fn component_ref_parts(reference: &str) -> Result<(String, String), String> {
    let tail = reference
        .strip_prefix("#/components/")
        .ok_or_else(|| format!("unsupported external or non-component reference {reference}"))?;
    let mut parts = tail.split('/');
    let kind = parts.next().filter(|value| !value.is_empty());
    let name = parts.next().filter(|value| !value.is_empty());
    match (kind, name, parts.next()) {
        (Some(kind), Some(name), None) => Ok((decode_json_pointer(kind), decode_json_pointer(name))),
        _ => Err(format!("invalid component reference {reference}")),
    }
}

/// Decode the two JSON pointer escape sequences.
fn decode_json_pointer(value: &str) -> String {
    value.replace("~1", "/").replace("~0", "~")
}

/// Escape a component name for use in a JSON pointer.
fn escape_json_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

/// Copy an optional root field without changing its value.
fn copy_root_field(source: &YamlMapping, target: &mut YamlMapping, name: &str) {
    if let Some(value) = yaml_get(source, name) {
        target.insert(YamlValue::String(name.to_owned()), value.clone());
    }
}

/// Get a string-keyed value from a YAML mapping.
fn yaml_get<'a>(mapping: &'a YamlMapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(&YamlValue::String(key.to_owned()))
}

/// Write a refreshed manifest and exact complete upstream source.
fn write_artifacts(manifest: &ReferenceManifest, source: &[u8]) -> Result<(), String> {
    let manifest_path = repo_root().join(OPENAI_REFERENCE_MANIFEST);
    let source_path = repo_root().join(OPENAI_REFERENCE_SPEC);
    let manifest_content = serde_json::to_string_pretty(manifest)
        .map(|content| format!("{content}\n"))
        .map_err(|e| format!("failed to serialize reference manifest: {e}"))?;

    std::fs::write(&manifest_path, manifest_content)
        .map_err(|e| format!("failed to write {}: {e}", manifest_path.display()))?;
    std::fs::write(&source_path, source).map_err(|e| format!("failed to write {}: {e}", source_path.display()))
}

/// Read one reference manifest.
fn read_manifest(path: &Path) -> Result<ReferenceManifest, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&content).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

/// Validate immutable pin and fixed source identity fields.
fn validate_manifest(manifest: &ReferenceManifest) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported reference manifest schema {}",
            manifest.schema_version
        ));
    }
    if manifest.repository != OPENAI_OPENAPI_REPOSITORY || manifest.path != OPENAI_OPENAPI_PATH {
        return Err("reference manifest must point to openai/openai-openapi/openapi.yaml".to_owned());
    }
    validate_revision(&manifest.revision)?;
    validate_sha256(&manifest.source_sha256)
}

/// Require a full Git commit rather than a mutable branch or tag.
fn validate_revision(revision: &str) -> Result<(), String> {
    if revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err("reference revision must be a 40-character hexadecimal Git commit".to_owned())
    }
}

/// Require a lowercase SHA-256 digest.
fn validate_sha256(digest: &str) -> Result<(), String> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err("source_sha256 must be a 64-character lowercase hexadecimal digest".to_owned())
    }
}

/// Compute a lowercase SHA-256 digest.
pub(super) fn sha256_hex(content: &[u8]) -> String {
    format!("{:x}", Sha256::digest(content))
}

/// Normalize generated text to one final newline.
fn ensure_trailing_newline(mut content: String) -> String {
    while content.ends_with("\n\n") {
        content.pop();
    }
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content
}
