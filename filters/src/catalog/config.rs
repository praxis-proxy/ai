// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the model catalog filter.

use std::collections::HashSet;

use praxis_filter::FilterError;
use serde::{Deserialize, Serialize};

/// Default path that serves the OpenAI-compatible model listing.
pub(super) const DEFAULT_LIST_PATH: &str = "/v1/models";

/// Default seconds between upstream model list refreshes.
const DEFAULT_REFRESH_INTERVAL_SECS: u64 = 60;

/// YAML config for the `model_catalog` filter.
///
/// ```yaml
/// filter: model_catalog
/// list_path: /v1/models          # default
/// default_visibility: public     # public | authenticated
/// refresh_interval_secs: 60      # default
/// unconfigured_models: show      # show | hide
/// upstreams:
///   - name: openai
///     url: https://api.openai.com/v1/models
///     provider: openai
///     bearer_token_env_var: OPENAI_API_KEY
///   - name: vllm
///     url: http://vllm.inference.svc:8000/v1/models
///     provider: vllm
///     allow_private: true
/// models:
///   - id: gpt-4o
///     version: "2024-08-06"
///     capabilities: [chat_completions, responses]
///     cost_tier: standard
///     aliases: [gpt4o]
///   - id: llama-70b
///     upstream_id: meta-llama/Llama-3.1-70B-Instruct
///     capabilities: [chat_completions]
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

    /// Upstream model list endpoints. When set, they decide which models
    /// exist and `models` entries only overlay metadata and visibility.
    #[serde(default)]
    pub upstreams: Vec<UpstreamConfig>,

    /// Seconds an upstream model list is served before it is refreshed.
    #[serde(default = "default_refresh_interval_secs")]
    pub refresh_interval_secs: u64,

    /// Whether upstream models without a `models` entry are listed.
    #[serde(default)]
    pub unconfigured_models: UnconfiguredModels,

    /// Catalog entries. IDs should be the client-facing model IDs that
    /// inference routing accepts. Required when `upstreams` is empty.
    #[serde(default)]
    pub models: Vec<ModelEntryConfig>,
}

/// One OpenAI-compatible `GET /v1/models` endpoint the catalog reads.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UpstreamConfig {
    /// Name used in logs.
    pub name: String,

    /// Full URL of the upstream model list endpoint.
    pub url: String,

    /// Provider reported in catalog metadata for models without an entry.
    #[serde(default)]
    pub provider: Option<String>,

    /// Environment variable holding a token sent as `Authorization: Bearer`.
    #[serde(default)]
    pub bearer_token_env_var: Option<String>,

    /// Extra request headers, such as `anthropic-version`.
    #[serde(default)]
    pub headers: Vec<UpstreamHeaderConfig>,

    /// Allow connecting to private, loopback, and link-local addresses.
    #[serde(default)]
    pub allow_private: bool,
}

/// One header sent with an upstream model list request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UpstreamHeaderConfig {
    /// Header name.
    pub name: String,

    /// Literal header value.
    #[serde(default)]
    pub value: Option<String>,

    /// Environment variable holding the header value.
    #[serde(default)]
    pub value_env_var: Option<String>,
}

/// Listing policy for upstream models that have no `models` entry.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum UnconfiguredModels {
    /// List them with `default_visibility` and the upstream's provider.
    #[default]
    Show,
    /// List only models that have an entry.
    Hide,
}

/// One model advertised by the catalog.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelEntryConfig {
    /// Client-facing model ID.
    pub id: String,

    /// ID the upstream reports for this model. Defaults to `id`. Only used
    /// with `upstreams`; the upstream ID itself is not listed.
    #[serde(default)]
    pub upstream_id: Option<String>,

    /// OpenAI `owned_by` value. Defaults to the upstream's value, then
    /// `provider`, then `praxis`.
    #[serde(default)]
    pub owned_by: Option<String>,

    /// Unix timestamp reported as `created`. Defaults to the upstream's
    /// value, then the top-level value.
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

impl ModelEntryConfig {
    /// ID matched against upstream model lists.
    pub(super) fn upstream_id(&self) -> &str {
        self.upstream_id.as_deref().unwrap_or(&self.id)
    }
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
    if cfg.upstreams.is_empty() {
        if cfg.models.is_empty() {
            return Err("model_catalog: models must not be empty when no upstreams are configured".into());
        }
        if let Some(entry) = cfg.models.iter().find(|entry| entry.upstream_id.is_some()) {
            return Err(format!("model_catalog: upstream_id for '{}' requires upstreams", entry.id).into());
        }
    } else {
        validate_upstreams(cfg)?;
    }
    validate_entries(&cfg.models)
}

/// Validate every entry and reject IDs claimed twice.
fn validate_entries(entries: &[ModelEntryConfig]) -> Result<(), FilterError> {
    let mut seen = HashSet::new();
    let mut upstream_ids = HashSet::new();
    for entry in entries {
        validate_entry(entry, &mut seen)?;
        if let Some(upstream_id) = &entry.upstream_id {
            validate_id(upstream_id)?;
        }
        if !upstream_ids.insert(entry.upstream_id()) {
            return Err(format!(
                "model_catalog: upstream id '{}' is claimed by more than one entry; use aliases instead",
                entry.upstream_id()
            )
            .into());
        }
    }
    Ok(())
}

/// Validate discovery settings and each upstream endpoint.
fn validate_upstreams(cfg: &ModelCatalogConfig) -> Result<(), FilterError> {
    if cfg.refresh_interval_secs == 0 {
        return Err("model_catalog: refresh_interval_secs must be at least 1".into());
    }
    if cfg.unconfigured_models == UnconfiguredModels::Hide && cfg.models.is_empty() {
        return Err("model_catalog: unconfigured_models: hide requires at least one models entry".into());
    }
    let mut names = HashSet::new();
    for upstream in &cfg.upstreams {
        if !names.insert(upstream.name.as_str()) {
            return Err(format!("model_catalog: duplicate upstream name '{}'", upstream.name).into());
        }
        validate_upstream(upstream)?;
    }
    Ok(())
}

/// Validate one upstream endpoint's settings.
fn validate_upstream(upstream: &UpstreamConfig) -> Result<(), FilterError> {
    if upstream.name.is_empty() {
        return Err("model_catalog: upstream names must not be empty".into());
    }
    if upstream.provider.as_deref().is_some_and(str::is_empty) {
        return Err(format!(
            "model_catalog: provider for upstream '{}' must not be empty",
            upstream.name
        )
        .into());
    }
    for header in &upstream.headers {
        if header.value.is_some() == header.value_env_var.is_some() {
            return Err(format!(
                "model_catalog: header '{}' for upstream '{}' needs exactly one of value or value_env_var",
                header.name, upstream.name
            )
            .into());
        }
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
pub(super) fn validate_id(id: &str) -> Result<(), FilterError> {
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

/// Serde default for `refresh_interval_secs`.
fn default_refresh_interval_secs() -> u64 {
    DEFAULT_REFRESH_INTERVAL_SECS
}
