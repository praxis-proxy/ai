// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the AI guardrails filter.

use praxis_core::config::ChainRef;
use serde::Deserialize;

/// Deserialized YAML config for the `ai_guardrails` filter.
///
/// ```yaml
/// filter: ai_guardrails
/// outbound_chain: nemo-outbound # optional; defaults to an empty chain
/// provider:
///   type: nemo
///   endpoint: "http://nemo:8000/v1/checks"
///   model: "check-model"
///   guardrails:
///     config_ids: ["your-config"]
///   timeout_ms: 5000
/// phase:
///   request: true
///   response: false
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AiGuardrailsConfig {
    /// Outbound filter chain executed for every `NeMo` callout.
    ///
    /// Optional. Callouts always run through the filtered-subrequest executor;
    /// omitting this field uses an empty inline chain (pure passthrough).
    #[serde(default = "default_outbound_chain")]
    pub outbound_chain: ChainRef,

    /// External provider configuration (required).
    pub provider: ProviderConfig,

    /// Which phases to evaluate.
    #[serde(default)]
    pub phase: PhaseConfig,
}

/// Default `outbound_chain` when the field is omitted: an empty inline chain.
fn default_outbound_chain() -> ChainRef {
    ChainRef::Inline {
        name: "ai_guardrails_outbound".to_owned(),
        filters: Vec::new(),
    }
}

/// Supported external guardrail provider types.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ProviderType {
    /// NVIDIA `NeMo` Guardrails via `/v1/checks`.
    Nemo,
}

/// Provider type selector and opaque provider-specific configuration.
///
/// The `type` field selects the provider. All remaining fields are
/// captured via `#[serde(flatten)]` and passed to the provider's
/// own `from_config` for parsing and validation.
#[derive(Debug, Deserialize)]
pub(crate) struct ProviderConfig {
    /// Provider type selector.
    #[serde(rename = "type")]
    pub provider_type: ProviderType,

    /// Provider-specific fields (parsed by each provider's `from_config`).
    #[serde(flatten)]
    pub config: serde_yaml::Value,
}

/// Controls which phases (request/response) the filter evaluates.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhaseConfig {
    /// Evaluate client requests before forwarding to the upstream.
    #[serde(default = "default_true")]
    pub request: bool,

    /// Evaluate upstream responses before forwarding to the client.
    #[serde(default)]
    pub response: bool,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            request: true,
            response: false,
        }
    }
}

/// Returns `true` for serde default fields.
fn default_true() -> bool {
    true
}
