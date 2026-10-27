// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the model catalog filter.

use std::collections::HashSet;

use praxis_filter::FilterError;
use serde::{Deserialize, Serialize};

/// Default path that serves the OpenAI-compatible model listing.
pub(super) const DEFAULT_LIST_PATH: &str = "/v1/models";

/// YAML config for the `model_catalog` filter.
///
/// ```yaml
/// filter: model_catalog
/// list_path: /v1/models          # default
/// default_visibility: public     # public | authenticated
/// models:
///   - id: gpt-4o
///     owned_by: openai
///     provider: openai
///     version: "2024-08-06"
///     capabilities: [chat_completions, responses]
///     cost_tier: standard
///     aliases: [gpt4o]
///   - id: claude-sonnet-4-5
///     owned_by: anthropic
///     provider: vertex
///     capabilities: [messages]
///     cost_tier: premium
///     visible_to:
///       subjects: [alice]
///       roles: [ml-eng]
///       teams: [research]
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelCatalogConfig {
    /// Path serving the model list. `{list_path}/{id}` serves one model.
    #[serde(default = "default_list_path")]
    pub list_path: String,

    /// Unix timestamp reported as `created` when an entry omits its own.
    #[serde(default)]
    pub created: u64,

    /// Who may see entries that have no `visible_to` rule.
    #[serde(default)]
    pub default_visibility: DefaultVisibility,

    /// Catalog entries in listing order. IDs should be the client-facing model
    /// IDs that inference routing accepts.
    pub models: Vec<ModelEntryConfig>,
}

/// One model advertised by the catalog.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelEntryConfig {
    /// Client-facing model ID.
    pub id: String,

    /// OpenAI `owned_by` value. Defaults to `provider`, then `praxis`.
    #[serde(default)]
    pub owned_by: Option<String>,

    /// Unix timestamp reported as `created`. Defaults to the top-level value.
    #[serde(default)]
    pub created: Option<u64>,

    /// Provider serving the model, reported in catalog metadata.
    #[serde(default)]
    pub provider: Option<String>,

    /// Model version, reported in catalog metadata.
    #[serde(default)]
    pub version: Option<String>,

    /// Operator-declared API capabilities. These are not verified upstream.
    #[serde(default)]
    pub capabilities: Vec<Capability>,

    /// Coarse cost tier label, reported in catalog metadata.
    #[serde(default)]
    pub cost_tier: Option<String>,

    /// Additional IDs listed as their own objects with the same visibility.
    #[serde(default)]
    pub aliases: Vec<String>,

    /// Restrict visibility to matching principals. Absent uses
    /// `default_visibility`.
    #[serde(default)]
    pub visible_to: Option<VisibleToConfig>,
}

/// Principals allowed to see an entry. A caller matching any list sees it.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct VisibleToConfig {
    /// Authenticated subject IDs.
    #[serde(default)]
    pub subjects: Vec<String>,

    /// Authenticated roles.
    #[serde(default)]
    pub roles: Vec<String>,

    /// Authenticated teams.
    #[serde(default)]
    pub teams: Vec<String>,
}

impl VisibleToConfig {
    /// Whether no principal could ever match this rule.
    fn is_empty(&self) -> bool {
        self.subjects.is_empty() && self.roles.is_empty() && self.teams.is_empty()
    }
}

/// Visibility of entries without a `visible_to` rule.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DefaultVisibility {
    /// Visible to every caller.
    #[default]
    Public,
    /// Visible only to callers with an authenticated identity.
    Authenticated,
}

/// API capability an operator declares for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Capability {
    /// `POST /v1/chat/completions`.
    ChatCompletions,
    /// `POST /v1/completions`.
    Completions,
    /// `POST /v1/responses`.
    Responses,
    /// `POST /v1/messages`.
    Messages,
    /// `POST /v1/embeddings`.
    Embeddings,
    /// `POST /v1/audio/speech`.
    AudioSpeech,
    /// `POST /v1/audio/transcriptions`.
    AudioTranscriptions,
    /// `POST /v1/images/generations`.
    Images,
    /// Realtime sessions.
    Realtime,
}

/// Validate config at construction time.
pub(super) fn validate_config(cfg: &ModelCatalogConfig) -> Result<(), FilterError> {
    if !cfg.list_path.starts_with('/') {
        return Err("model_catalog: list_path must start with '/'".into());
    }
    if cfg.list_path.len() > 1 && cfg.list_path.ends_with('/') {
        return Err("model_catalog: list_path must not end with '/'".into());
    }
    if cfg.models.is_empty() {
        return Err("model_catalog: models must not be empty".into());
    }

    let mut seen = HashSet::new();
    for entry in &cfg.models {
        validate_entry(entry, &mut seen)?;
    }
    Ok(())
}

/// Validate one entry, recording its IDs in `seen` to reject duplicates.
fn validate_entry<'a>(entry: &'a ModelEntryConfig, seen: &mut HashSet<&'a str>) -> Result<(), FilterError> {
    for id in std::iter::once(&entry.id).chain(&entry.aliases) {
        validate_id(id)?;
        if !seen.insert(id.as_str()) {
            return Err(format!("model_catalog: duplicate model id '{id}'").into());
        }
    }
    if entry.visible_to.as_ref().is_some_and(VisibleToConfig::is_empty) {
        return Err(format!(
            "model_catalog: visible_to for '{}' must list at least one subject, role, or team",
            entry.id
        )
        .into());
    }
    for (field, value) in [
        ("owned_by", &entry.owned_by),
        ("provider", &entry.provider),
        ("version", &entry.version),
        ("cost_tier", &entry.cost_tier),
    ] {
        if value.as_deref().is_some_and(str::is_empty) {
            return Err(format!("model_catalog: {field} for '{}' must not be empty", entry.id).into());
        }
    }
    Ok(())
}

/// Reject IDs that cannot round-trip through `{list_path}/{id}`.
fn validate_id(id: &str) -> Result<(), FilterError> {
    if id.is_empty() {
        return Err("model_catalog: model ids must not be empty".into());
    }
    if id
        .chars()
        .any(|c| c.is_control() || c.is_whitespace() || c == '?' || c == '#')
    {
        return Err(format!("model_catalog: model id '{id}' contains a character that cannot appear in a path").into());
    }
    Ok(())
}

/// Serde default for `list_path`.
fn default_list_path() -> String {
    DEFAULT_LIST_PATH.to_owned()
}
