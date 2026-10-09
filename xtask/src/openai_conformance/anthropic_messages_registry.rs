// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Drift check between the runtime Anthropic Messages registry and the pinned spec.
//!
//! The registry is the runtime source of truth for Anthropic Messages operation
//! identity. This check fails when a registered operation's method, path, or
//! operation ID no longer agrees with the pinned Anthropic specification.

use serde::Deserialize;

use super::{
    area::{ANTHROPIC_REFERENCE_MANIFEST, ANTHROPIC_REFERENCE_SPEC},
    model::{OperationScope, SpecOperation},
    reference::sha256_hex,
    registry_check::{Comparison, compare},
    spec::{parse_openapi_operations, read_spec, repo_root, scope_operations},
};

/// Anthropic Messages operations selected from the pinned specification.
const ANTHROPIC_MESSAGES_SCOPE: OperationScope =
    OperationScope::new("anthropic_messages", "Anthropic Messages", &["/v1/messages"]);

/// Provenance manifest for the vendored Anthropic specification.
#[derive(Deserialize)]
struct AnthropicManifest {
    /// Manifest schema version.
    schema_version: u32,
    /// SHA-256 of the decompressed vendored JSON.
    vendored_sha256: String,
}

/// Compare the runtime Anthropic Messages registry against the pinned specification.
pub(super) fn check() -> Result<String, String> {
    let content = read_spec(ANTHROPIC_REFERENCE_SPEC)?;
    verify_provenance(&content)?;

    let all_operations = parse_openapi_operations(&content)?;
    let scoped = scope_operations(all_operations, ANTHROPIC_MESSAGES_SCOPE);
    let non_beta: Vec<_> = scoped.into_iter().filter(|op| !op.beta).collect();

    if non_beta.is_empty() {
        return Err(format!(
            "pinned Anthropic spec {ANTHROPIC_REFERENCE_SPEC} did not contain any non-beta Messages operations"
        ));
    }

    compare_registry(&non_beta)
}

/// Compare every registered operation against the projected specification.
fn compare_registry(spec_operations: &[SpecOperation]) -> Result<String, String> {
    let registry = praxis_ai_apis::anthropic::routes::operation_specs();
    let mut checked: usize = 0;
    let mut failures = Vec::new();

    for spec in registry {
        checked += 1;
        let found = spec_operations
            .iter()
            .find(|candidate| {
                candidate.key.method == spec.method().as_str() && candidate.key.path == spec.runtime_path()
            })
            .and_then(|candidate| candidate.operation_id.as_deref());

        if let Comparison::Drifted(reason) = compare(
            spec.method().as_str(),
            spec.runtime_path(),
            spec.operation_id(),
            found,
            false,
        ) {
            failures.push(reason);
        }
    }

    if failures.is_empty() {
        Ok(format!(
            "anthropic messages registry matches the pinned specification: {checked} operations checked"
        ))
    } else {
        Err(failures.join("\n"))
    }
}

/// Verify the vendored spec digest against the provenance manifest.
fn verify_provenance(content: &str) -> Result<(), String> {
    let manifest_path = repo_root().join(ANTHROPIC_REFERENCE_MANIFEST);
    let manifest_content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("failed to read {}: {e}", manifest_path.display()))?;
    let manifest: AnthropicManifest = serde_json::from_str(&manifest_content)
        .map_err(|e| format!("failed to parse {}: {e}", manifest_path.display()))?;

    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported Anthropic manifest schema {}",
            manifest.schema_version
        ));
    }

    let digest = sha256_hex(content.as_bytes());
    if digest != manifest.vendored_sha256 {
        return Err(format!(
            "vendored Anthropic spec digest mismatch: expected {}, got {digest}",
            manifest.vendored_sha256,
        ));
    }

    Ok(())
}
