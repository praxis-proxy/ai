// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the Anthropic-to-Chat-Completions transformation filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum request body size (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// LossyFeature
// -----------------------------------------------------------------------------

/// An Anthropic feature the operator may allow the translator to degrade.
///
/// Absent from the allowlist, each of these keeps the strict reject behavior:
/// a request that uses the feature is answered with a 400 before any backend
/// call. Listed, the translator removes the feature's wire markers (after
/// validating their shape), reports the degradation to the operator, and
/// completes the translation so an unmodified client can still drive the
/// Chat Completions backend.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LossyFeature {
    /// Anthropic extended thinking: the `thinking` request field and the
    /// thinking-only `context_management` edits that accompany it. Degrading
    /// removes them; the translated response cannot carry thinking blocks.
    ExtendedThinking,

    /// Anthropic prompt caching: `cache_control` markers on system, message,
    /// tool-result, and tool blocks. Degrading removes the markers while
    /// preserving the prompt and tool content; explicit cache breakpoints are
    /// not honored, so cost and latency may differ.
    PromptCaching,
}

// -----------------------------------------------------------------------------
// LossyFeatureAllowlist
// -----------------------------------------------------------------------------

/// The resolved set of features an operator opted into degrading.
///
/// A flat, `Copy` view of [`LossyFeature`] entries so the request translator
/// can branch on each feature without re-scanning the configured list.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LossyFeatureAllowlist {
    /// Degrade Anthropic extended thinking (`thinking`, `context_management`).
    pub extended_thinking: bool,
    /// Degrade Anthropic prompt caching (`cache_control`).
    pub prompt_caching: bool,
}

impl LossyFeatureAllowlist {
    /// Whether any lossy feature is allowed.
    pub(crate) fn any(self) -> bool {
        self.extended_thinking || self.prompt_caching
    }
}

// -----------------------------------------------------------------------------
// AnthropicMessagesToChatCompletionsConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`AnthropicMessagesToChatCompletionsFilter`].
///
/// [`AnthropicMessagesToChatCompletionsFilter`]: super::AnthropicMessagesToChatCompletionsFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicMessagesToChatCompletionsConfig {
    /// Anthropic features the operator allows the translator to degrade.
    ///
    /// Empty by default, which keeps the strict reject behavior for every
    /// feature. Unknown names fail config loading (`deny_unknown_fields` plus
    /// the validated `LossyFeature` enum), so a typo cannot silently widen
    /// what the translator will drop.
    #[serde(default)]
    pub allow_lossy_features: Vec<LossyFeature>,

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

impl AnthropicMessagesToChatCompletionsConfig {
    /// Resolve the configured list into a flat, `Copy` allowlist.
    pub(crate) fn allowlist(&self) -> LossyFeatureAllowlist {
        let mut allow = LossyFeatureAllowlist::default();
        for feature in &self.allow_lossy_features {
            match feature {
                LossyFeature::ExtendedThinking => allow.extended_thinking = true,
                LossyFeature::PromptCaching => allow.prompt_caching = true,
            }
        }
        allow
    }
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(
    cfg: AnthropicMessagesToChatCompletionsConfig,
) -> Result<AnthropicMessagesToChatCompletionsConfig, FilterError> {
    validate_max_body_bytes("anthropic_messages_to_chat_completions", cfg.max_body_bytes)?;
    Ok(cfg)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn absent_allowlist_degrades_nothing() {
        let cfg = parse("max_body_bytes: 1024").unwrap();
        assert_eq!(cfg.allowlist(), LossyFeatureAllowlist::default());
        assert!(!cfg.allowlist().any());
    }

    #[test]
    fn empty_allowlist_degrades_nothing() {
        let cfg = parse("allow_lossy_features: []").unwrap();
        assert!(!cfg.allowlist().any());
    }

    #[test]
    fn allowlist_resolves_each_feature() {
        let cfg = parse("allow_lossy_features:\n  - prompt_caching\n  - extended_thinking").unwrap();
        assert_eq!(
            cfg.allowlist(),
            LossyFeatureAllowlist {
                prompt_caching: true,
                extended_thinking: true,
            }
        );
    }

    #[test]
    fn allowlist_accepts_a_single_feature() {
        let cfg = parse("allow_lossy_features:\n  - prompt_caching").unwrap();
        assert_eq!(
            cfg.allowlist(),
            LossyFeatureAllowlist {
                prompt_caching: true,
                extended_thinking: false,
            }
        );
    }

    #[test]
    fn duplicate_entries_collapse() {
        let cfg = parse("allow_lossy_features:\n  - prompt_caching\n  - prompt_caching").unwrap();
        assert!(cfg.allowlist().prompt_caching);
        assert!(!cfg.allowlist().extended_thinking);
    }

    #[test]
    fn unknown_feature_name_fails_config_loading() {
        let error = parse("allow_lossy_features:\n  - vision").unwrap_err();
        assert!(
            error.to_string().contains("vision") || error.to_string().contains("unknown variant"),
            "unknown feature name must fail config loading: {error}"
        );
    }

    #[test]
    fn unknown_top_level_field_fails_config_loading() {
        let error = parse("allow_lossy: [prompt_caching]").unwrap_err();
        assert!(
            error.to_string().contains("allow_lossy") || error.to_string().contains("unknown field"),
            "unknown top-level field must fail config loading: {error}"
        );
    }

    // Test Utilities

    fn parse(yaml: &str) -> Result<AnthropicMessagesToChatCompletionsConfig, FilterError> {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let cfg = praxis_filter::parse_filter_config("anthropic_messages_to_chat_completions", &value)?;
        build_config(cfg)
    }
}
