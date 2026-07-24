// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Static site/capability descriptor model for gateway-to-gateway routing.
//!
//! Defines the local routing records consumed by the `grid_route`
//! filter. Records are validated at parse time and read immutably
//! during request handling.
//!
//! This module defines the data model only. Scoring and route
//! extraction logic live in the `route` sibling module.

use std::{collections::HashSet, sync::Arc};

use praxis_filter::FilterError;
use serde::Deserialize;

/// Maximum number of route candidates.
const MAX_CANDIDATES: usize = 1024;

/// Maximum length for identifier strings.
const MAX_NAME_LEN: usize = 256;

/// Header prefixes reserved for internal gateway/protocol metadata.
pub(crate) const RESERVED_HEADER_PREFIXES: &[&str] = &["x-praxis-", "x-mcp-"];

// -----------------------------------------------------------------------------
// CapabilityKind
// -----------------------------------------------------------------------------

/// Capability kind for descriptor matching.
///
/// Categorises what a route candidate offers. The route filter
/// uses this to decide which request metadata can match a candidate.
///
/// `InferenceModel` is matched by the model request header.
/// `McpTool` is matched by `mcp.method`=`tools/call` + `mcp.name`
/// metadata, which takes precedence over the model header.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapabilityKind {
    /// OpenAI-compatible inference model.
    InferenceModel,

    /// MCP tool, matched by `mcp.method`=`tools/call` + `mcp.name` metadata.
    McpTool,
}

impl CapabilityKind {
    /// Short string for diagnostics and route metadata.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::InferenceModel => "inference_model",
            Self::McpTool => "mcp_tool",
        }
    }
}

// -----------------------------------------------------------------------------
// Credential reference
// -----------------------------------------------------------------------------

/// Reference to a Kubernetes Secret that holds a bearer token.
///
/// Contains only locating information — never the credential value itself.
/// Safe to persist in a `ConfigMap`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CredentialRef {
    /// Secret name.
    pub name: String,
    /// Secret namespace.
    pub namespace: String,
    /// Key within `Secret.data` that holds the credential bytes.
    pub key: String,
}

/// Credential reference projected alongside a route candidate.
///
/// Identifies the authentication strategy and the Kubernetes Secret that
/// holds the bearer token.  The filter carries this reference for downstream
/// consumers; it does not perform Kubernetes API calls or inject the credential
/// at request time.
///
/// Field names use camelCase to match the operator's overlay serialization.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub(crate) struct CandidateCredential {
    /// Authentication strategy identifier.  Currently `"bearer_token"`.
    pub strategy: String,

    /// Reference to the Kubernetes Secret that contains the token.
    #[serde(rename = "secretRef")]
    pub secret_ref: CredentialRef,
}

// -----------------------------------------------------------------------------
// CandidateConfig (serde)
// -----------------------------------------------------------------------------

/// A single route candidate as written in YAML config.
///
/// ```yaml
/// candidates:
///   - kind: inference_model
///     name: llama-3.1-8b
///     site: site-b
///     cluster: grid-site-b
///   - kind: mcp_tool
///     name: weather-lookup
///     site: site-c
///     cluster: grid-site-c
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CandidateConfig {
    /// Cluster name to select when this candidate is chosen.
    pub cluster: String,

    /// Whether this candidate is fresh (default: `true`).
    #[serde(default = "default_fresh")]
    pub fresh: bool,

    /// Capability kind.
    pub kind: CapabilityKind,

    /// Capability name (model name, tool name, or agent name).
    pub name: String,

    /// Site that owns this capability.
    pub site: String,

    /// Optional credential reference for upstream authentication.
    ///
    /// When present, the upstream identified by this candidate requires a bearer
    /// token obtained from the referenced Kubernetes Secret.  The filter carries
    /// this reference for downstream use; it does not resolve the token.
    #[serde(default)]
    pub credential: Option<CandidateCredential>,
}

/// Default freshness state for candidates.
fn default_fresh() -> bool {
    true
}

// -----------------------------------------------------------------------------
// RouteCandidate (validated)
// -----------------------------------------------------------------------------

/// A validated route candidate ready for runtime matching.
///
/// Created by [`validate_candidates`] from raw config entries.
/// All string fields are bounded and non-blank.
#[derive(Debug)]
pub(crate) struct RouteCandidate {
    /// Cluster name to select.
    pub cluster: Arc<str>,

    /// Optional credential reference for upstream authentication.
    ///
    /// Carried from the operator overlay; not used by routing logic.
    pub credential: Option<CandidateCredential>,

    /// Whether this candidate is fresh.
    pub fresh: bool,

    /// Capability kind.
    pub kind: CapabilityKind,

    /// Capability name.
    pub name: Arc<str>,

    /// Site that owns this capability.
    pub site: Arc<str>,
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate and build the candidate list from raw config entries.
///
/// # Errors
///
/// Returns [`FilterError`] if:
/// - the candidate list is empty or exceeds [`MAX_CANDIDATES`]
/// - any name/site/cluster/credential field is blank or oversized
/// - duplicate (kind, name, site) tuples exist
pub(crate) fn validate_candidates(raw: Vec<CandidateConfig>) -> Result<Vec<RouteCandidate>, FilterError> {
    if raw.is_empty() {
        return Err("grid: candidates list must not be empty".into());
    }
    if raw.len() > MAX_CANDIDATES {
        return Err(format!("grid: candidates exceeds maximum of {MAX_CANDIDATES}").into());
    }

    let mut candidates = Vec::with_capacity(raw.len());
    let mut seen: HashSet<(CapabilityKind, String, String)> = HashSet::with_capacity(raw.len());

    for (i, c) in raw.into_iter().enumerate() {
        validate_name(&format!("candidates[{i}].name"), &c.name)?;
        validate_name(&format!("candidates[{i}].site"), &c.site)?;
        validate_name(&format!("candidates[{i}].cluster"), &c.cluster)?;
        validate_credential(i, c.credential.as_ref())?;

        if !seen.insert((c.kind, c.name.clone(), c.site.clone())) {
            return Err(format!("grid: duplicate candidate '{}/{}/{}'", c.kind.as_str(), c.name, c.site).into());
        }

        candidates.push(RouteCandidate {
            cluster: Arc::from(c.cluster.as_str()),
            credential: c.credential,
            fresh: c.fresh,
            kind: c.kind,
            name: Arc::from(c.name.as_str()),
            site: Arc::from(c.site.as_str()),
        });
    }

    Ok(candidates)
}

/// Validate the promoted model header name.
///
/// Rejects blank, unparseable, or reserved-prefix header names.
pub(crate) fn validate_model_header(raw: &str) -> Result<http::header::HeaderName, FilterError> {
    if raw.trim().is_empty() {
        return Err("grid: model_header must not be empty".into());
    }
    let header: http::header::HeaderName = raw
        .parse()
        .map_err(|e| -> FilterError { format!("grid: invalid model_header: {e}").into() })?;
    if RESERVED_HEADER_PREFIXES.iter().any(|p| header.as_str().starts_with(p)) {
        return Err("grid: model_header must not use a reserved internal header prefix".into());
    }
    Ok(header)
}

/// Validate a local site identifier.
pub(crate) fn validate_local_site(value: &str) -> Result<(), FilterError> {
    validate_name("local_site", value)
}

/// Validate a bounded, non-blank identifier.
fn validate_name(field: &str, value: &str) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > MAX_NAME_LEN {
        return Err(format!("grid: {field} must be 1-{MAX_NAME_LEN} non-blank characters").into());
    }
    Ok(())
}

/// Validate optional credential reference fields.
fn validate_credential(index: usize, credential: Option<&CandidateCredential>) -> Result<(), FilterError> {
    let Some(credential) = credential else {
        return Ok(());
    };
    validate_name(
        &format!("candidates[{index}].credential.strategy"),
        &credential.strategy,
    )?;
    validate_name(
        &format!("candidates[{index}].credential.secretRef.name"),
        &credential.secret_ref.name,
    )?;
    validate_name(
        &format!("candidates[{index}].credential.secretRef.namespace"),
        &credential.secret_ref.namespace,
    )?;
    validate_name(
        &format!("candidates[{index}].credential.secretRef.key"),
        &credential.secret_ref.key,
    )
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;

    // ---- Valid configs ----

    #[test]
    fn valid_minimal_inference_candidate() {
        let candidates =
            validate_candidates(vec![candidate("inference_model", "llama", "site-a", "gateway-a")]).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].kind, CapabilityKind::InferenceModel);
        assert_eq!(&*candidates[0].name, "llama");
        assert_eq!(&*candidates[0].site, "site-a");
        assert_eq!(&*candidates[0].cluster, "gateway-a");
        assert!(candidates[0].fresh);
    }

    #[test]
    fn multiple_candidates_accepted() {
        let candidates = validate_candidates(vec![
            candidate("inference_model", "llama", "site-a", "c1"),
            candidate("inference_model", "granite", "site-b", "c2"),
        ])
        .unwrap();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn fresh_defaults_to_true() {
        let candidates = validate_candidates(vec![candidate("inference_model", "m", "s", "c")]).unwrap();
        assert!(candidates[0].fresh);
    }

    #[test]
    fn stale_candidate_preserved() {
        let mut c = candidate("inference_model", "m", "s", "c");
        c.fresh = false;
        let candidates = validate_candidates(vec![c]).unwrap();
        assert!(!candidates[0].fresh);
    }

    // ---- Rejections ----

    #[test]
    fn empty_candidates_rejected() {
        let err = validate_candidates(vec![]).expect_err("should fail");
        assert!(err.to_string().contains("must not be empty"), "{err}");
    }

    #[test]
    fn blank_name_rejected() {
        let err = validate_candidates(vec![candidate("inference_model", "", "s", "c")]).expect_err("should fail");
        assert!(err.to_string().contains("name must be"), "{err}");
    }

    #[test]
    fn whitespace_name_rejected() {
        let err = validate_candidates(vec![candidate("inference_model", "   ", "s", "c")]).expect_err("should fail");
        assert!(err.to_string().contains("name must be"), "{err}");
    }

    #[test]
    fn blank_site_rejected() {
        let err = validate_candidates(vec![candidate("inference_model", "m", "", "c")]).expect_err("should fail");
        assert!(err.to_string().contains("site must be"), "{err}");
    }

    #[test]
    fn blank_cluster_rejected() {
        let err = validate_candidates(vec![candidate("inference_model", "m", "s", "")]).expect_err("should fail");
        assert!(err.to_string().contains("cluster must be"), "{err}");
    }

    #[test]
    fn oversized_name_rejected() {
        let long = "a".repeat(MAX_NAME_LEN + 1);
        let err = validate_candidates(vec![candidate("inference_model", &long, "s", "c")]).expect_err("should fail");
        assert!(err.to_string().contains("name must be"), "{err}");
    }

    #[test]
    fn oversized_site_rejected() {
        let long = "a".repeat(MAX_NAME_LEN + 1);
        let err = validate_candidates(vec![candidate("inference_model", "m", &long, "c")]).expect_err("should fail");
        assert!(err.to_string().contains("site must be"), "{err}");
    }

    #[test]
    fn oversized_cluster_rejected() {
        let long = "a".repeat(MAX_NAME_LEN + 1);
        let err = validate_candidates(vec![candidate("inference_model", "m", "s", &long)]).expect_err("should fail");
        assert!(err.to_string().contains("cluster must be"), "{err}");
    }

    #[test]
    fn duplicate_candidate_rejected() {
        let err = validate_candidates(vec![
            candidate("inference_model", "llama", "site-a", "c1"),
            candidate("inference_model", "llama", "site-a", "c2"),
        ])
        .expect_err("should fail");
        assert!(err.to_string().contains("duplicate candidate"), "{err}");
    }

    #[test]
    fn same_name_different_site_not_duplicate() {
        let result = validate_candidates(vec![
            candidate("inference_model", "llama", "site-a", "c1"),
            candidate("inference_model", "llama", "site-b", "c2"),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn same_model_different_site_not_duplicate() {
        let result = validate_candidates(vec![
            candidate("inference_model", "llama", "site-a", "c1"),
            candidate("inference_model", "llama", "site-b", "c2"),
        ]);
        assert!(result.is_ok(), "same model on different sites is not a duplicate");
    }

    #[test]
    fn unknown_capability_kind_rejected() {
        let yaml = "- kind: unknown_thing\n  name: x\n  site: s\n  cluster: c";
        let err: Result<Vec<CandidateConfig>, _> = serde_yaml::from_str(yaml);
        assert!(err.is_err(), "unknown capability kind should be rejected by serde");
    }

    #[test]
    fn mcp_tool_candidate_valid() {
        let candidates = validate_candidates(vec![candidate("mcp_tool", "weather", "site-b", "gateway-b")]).unwrap();
        assert_eq!(candidates[0].kind, CapabilityKind::McpTool, "mcp_tool should parse");
        assert_eq!(&*candidates[0].name, "weather");
    }

    #[test]
    fn unsupported_capability_kind_rejected() {
        let yaml = "- kind: unsupported_kind\n  name: x\n  site: s\n  cluster: c";
        let err: Result<Vec<CandidateConfig>, _> = serde_yaml::from_str(yaml);
        assert!(err.is_err(), "unsupported capability kind is not supported");
    }

    #[test]
    fn mcp_and_inference_candidates_can_coexist() {
        let candidates = validate_candidates(vec![
            candidate("inference_model", "llama", "site-a", "c1"),
            candidate("mcp_tool", "weather", "site-b", "c2"),
        ])
        .unwrap();
        assert_eq!(candidates.len(), 2, "inference and mcp candidates can coexist");
    }

    #[test]
    fn deny_unknown_fields_on_candidate() {
        let yaml = "- kind: inference_model\n  name: x\n  site: s\n  cluster: c\n  extra: bad";
        let err: Result<Vec<CandidateConfig>, _> = serde_yaml::from_str(yaml);
        assert!(err.is_err(), "unknown fields should be rejected");
    }

    // ---- Credential reference ----

    #[test]
    fn credential_absent_defaults_to_none() {
        let candidates = validate_candidates(vec![candidate("inference_model", "m", "s", "c")]).unwrap();
        assert!(
            candidates[0].credential.is_none(),
            "absent credential must default to None"
        );
    }

    #[test]
    fn credential_bearing_candidate_parsed() {
        // Operator serializes ProjectedCredential with camelCase (secretRef).
        let yaml = concat!(
            "- kind: inference_model\n  name: m\n  site: s\n  cluster: c\n",
            "  credential:\n",
            "    strategy: bearer_token\n",
            "    secretRef:\n",
            "      name: my-secret\n",
            "      namespace: default\n",
            "      key: token\n",
        );
        let configs: Vec<CandidateConfig> = serde_yaml::from_str(yaml).unwrap();
        let cred = configs[0].credential.as_ref().unwrap();
        assert_eq!(cred.strategy, "bearer_token", "strategy must parse");
        assert_eq!(cred.secret_ref.name, "my-secret", "secretRef.name must parse");
        assert_eq!(cred.secret_ref.namespace, "default", "secretRef.namespace must parse");
        assert_eq!(cred.secret_ref.key, "token", "secretRef.key must parse");
    }

    #[test]
    fn unknown_credential_field_rejected() {
        let yaml = concat!(
            "- kind: inference_model\n  name: m\n  site: s\n  cluster: c\n",
            "  credential:\n",
            "    strategy: bearer_token\n",
            "    unexpected: nope\n",
            "    secretRef:\n",
            "      name: sec\n",
            "      namespace: ns\n",
            "      key: k\n",
        );
        let err = serde_yaml::from_str::<Vec<CandidateConfig>>(yaml);
        assert!(err.is_err(), "unknown credential fields must be rejected");
    }

    #[test]
    fn unknown_secret_ref_field_rejected() {
        let yaml = concat!(
            "- kind: inference_model\n  name: m\n  site: s\n  cluster: c\n",
            "  credential:\n",
            "    strategy: bearer_token\n",
            "    secretRef:\n",
            "      name: sec\n",
            "      namespace: ns\n",
            "      key: k\n",
            "      tokenValue: must-not-parse\n",
        );
        let err = serde_yaml::from_str::<Vec<CandidateConfig>>(yaml);
        assert!(err.is_err(), "unknown secretRef fields must be rejected");
    }

    #[test]
    fn blank_credential_secret_ref_field_rejected() {
        let yaml = concat!(
            "- kind: inference_model\n  name: m\n  site: s\n  cluster: c\n",
            "  credential:\n",
            "    strategy: bearer_token\n",
            "    secretRef:\n",
            "      name: sec\n",
            "      namespace: '   '\n",
            "      key: k\n",
        );
        let configs: Vec<CandidateConfig> = serde_yaml::from_str(yaml).unwrap();
        let err = validate_candidates(configs).unwrap_err();
        assert!(
            err.to_string().contains("credential.secretRef.namespace"),
            "blank credential namespace should be rejected with field context: {err}"
        );
    }

    #[test]
    fn oversized_credential_secret_ref_field_rejected() {
        let long = "a".repeat(MAX_NAME_LEN + 1);
        let yaml = format!(
            "- kind: inference_model\n  name: m\n  site: s\n  cluster: c\n\
             \x20 credential:\n\
             \x20   strategy: bearer_token\n\
             \x20   secretRef:\n\
             \x20     name: sec\n\
             \x20     namespace: ns\n\
             \x20     key: {long}\n"
        );
        let configs: Vec<CandidateConfig> = serde_yaml::from_str(&yaml).unwrap();
        let err = validate_candidates(configs).unwrap_err();
        assert!(
            err.to_string().contains("credential.secretRef.key"),
            "oversized credential key should be rejected with field context: {err}"
        );
    }

    #[test]
    fn credential_threads_through_validation() {
        let yaml = concat!(
            "- kind: inference_model\n  name: m\n  site: s\n  cluster: c\n",
            "  credential:\n",
            "    strategy: bearer_token\n",
            "    secretRef:\n",
            "      name: sec\n",
            "      namespace: ns\n",
            "      key: k\n",
        );
        let configs: Vec<CandidateConfig> = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_candidates(configs).unwrap();
        let cred = validated[0].credential.as_ref().unwrap();
        assert_eq!(
            cred.strategy, "bearer_token",
            "credential must survive validate_candidates"
        );
        assert_eq!(
            cred.secret_ref.name, "sec",
            "secretRef.name must survive validate_candidates"
        );
    }

    #[test]
    fn credential_does_not_affect_routing_fields() {
        let yaml = concat!(
            "- kind: inference_model\n  name: api-model\n  site: site-b\n  cluster: api-cluster\n",
            "  credential:\n",
            "    strategy: bearer_token\n",
            "    secretRef:\n",
            "      name: sec\n",
            "      namespace: ns\n",
            "      key: k\n",
        );
        let configs: Vec<CandidateConfig> = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_candidates(configs).unwrap();
        assert_eq!(&*validated[0].name, "api-model", "name must be preserved");
        assert_eq!(&*validated[0].site, "site-b", "site must be preserved");
        assert_eq!(&*validated[0].cluster, "api-cluster", "cluster must be preserved");
        assert!(validated[0].fresh, "fresh must default to true");
    }

    // ---- Model header validation ----

    #[test]
    fn valid_model_header() {
        assert!(validate_model_header("X-Model").is_ok());
    }

    #[test]
    fn blank_model_header_rejected() {
        let err = validate_model_header("").expect_err("should fail");
        assert!(err.to_string().contains("must not be empty"), "{err}");
    }

    #[test]
    fn reserved_prefix_model_header_rejected() {
        let err = validate_model_header("x-praxis-model").expect_err("should fail");
        assert!(err.to_string().contains("reserved"), "{err}");
    }

    // ---- Local site validation ----

    #[test]
    fn valid_local_site() {
        assert!(validate_local_site("site-a").is_ok());
    }

    #[test]
    fn blank_local_site_rejected() {
        let err = validate_local_site("").expect_err("should fail");
        assert!(err.to_string().contains("local_site must be"), "{err}");
    }

    // ---- Test utilities ----

    fn candidate(kind_str: &str, name: &str, site: &str, cluster: &str) -> CandidateConfig {
        let kind: CapabilityKind = serde_yaml::from_str(&format!("\"{kind_str}\"")).unwrap();
        CandidateConfig {
            cluster: cluster.to_owned(),
            credential: None,
            fresh: true,
            kind,
            name: name.to_owned(),
            site: site.to_owned(),
        }
    }
}
