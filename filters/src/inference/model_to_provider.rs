// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Resolves stable client-facing model IDs to a provider and provider target model.
//!
//! The exact configured model/path pair sets the internal
//! `x-praxis-ai-provider` route selector, rewrites the request body's `model`
//! field, and records the client-facing model ID in filter metadata for the
//! provider response adapter. Unknown models and paths pass through unchanged.
//! Configuration is local to the filter pipeline; this filter does not watch
//! Kubernetes resources or perform request-time control-plane lookups.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use std::collections::HashMap;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderName, HeaderValue};
use praxis_ai_apis::{MODEL_PROVIDER_CLIENT_MODEL_METADATA, MODEL_PROVIDER_HEADER, json_body::replace_json_body};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::Deserialize;
use serde_json::Value;

/// Maximum mappings accepted in one filter config.
const MAX_MODEL_MAPPINGS: usize = 1024;
/// Maximum paths accepted for one model mapping.
const MAX_PATHS_PER_MAPPING: usize = 16;
/// Maximum byte length for model and provider identifiers.
const MAX_MAPPING_VALUE_LEN: usize = 253;
/// Internal route-selection header written by this filter.
const PROVIDER_HEADER: HeaderName = HeaderName::from_static(MODEL_PROVIDER_HEADER);

/// Configures a client model's provider route and upstream model identity.
///
/// ```yaml
/// filter: model_to_provider
/// models:
///   - model: claude-sonnet-4-5
///     provider: vertex
///     target_model: vertex/claude-sonnet-4-5
///     paths: [/v1/messages, /v1/messages/count_tokens]
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelToProviderConfig {
    /// Exact client-facing model mappings.
    models: Vec<ModelProviderMappingConfig>,
    /// Maximum JSON request body buffered to inspect the model field.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// One public-model to provider mapping.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelProviderMappingConfig {
    /// Exact model ID sent by the client.
    model: String,
    /// Provider selector consumed by the unified `router` configuration.
    provider: String,
    /// Provider-specific model written into the request body.
    target_model: String,
    /// Exact inference paths on which this mapping applies.
    paths: Vec<String>,
}

/// Validated mapping used on the request path.
#[derive(Debug)]
struct ModelProviderMapping {
    /// Provider selector written to the internal route header.
    provider: HeaderValue,
    /// Provider-specific request-body model.
    target_model: String,
    /// Exact request paths accepted for this client model.
    paths: Vec<String>,
}

/// Maps stable client-facing model IDs to an internal provider selector and
/// provider-specific target model.
///
/// An exact model/path match sets `x-praxis-ai-provider` for the `router`,
/// rewrites the JSON body model, and records the client ID in filter metadata
/// for provider response adapters. Unknown models and paths pass through.
/// Configuration is pipeline-local: the filter performs no Kubernetes or
/// other control-plane lookups.
///
/// # YAML configuration
///
/// ```yaml
/// filter: model_to_provider
/// models:
///   - model: claude-sonnet-4-5
///     provider: vertex
///     target_model: vertex/claude-sonnet-4-5
///     paths: [/v1/messages, /v1/messages/count_tokens]
/// ```
pub struct ModelToProviderFilter {
    /// Exact public model ID to provider mapping table.
    models: HashMap<String, ModelProviderMapping>,
    /// Maximum buffered request body size.
    max_body_bytes: usize,
}

impl ModelToProviderFilter {
    /// Construct from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, duplicate, or invalid mappings and for an
    /// invalid request-body limit.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: ModelToProviderConfig = parse_filter_config("model_to_provider", config)?;
        if config.models.is_empty() || config.models.len() > MAX_MODEL_MAPPINGS {
            return Err(format!("model_to_provider: models must contain 1-{MAX_MODEL_MAPPINGS} mappings").into());
        }
        validate_max_body_bytes("model_to_provider", config.max_body_bytes)?;
        Ok(Box::new(Self {
            models: build_model_mappings(config.models)?,
            max_body_bytes: config.max_body_bytes,
        }))
    }
}

/// Validate and index the configured public model mappings.
fn build_model_mappings(
    entries: Vec<ModelProviderMappingConfig>,
) -> Result<HashMap<String, ModelProviderMapping>, FilterError> {
    let mut models = HashMap::with_capacity(entries.len());
    for entry in entries {
        let model = entry.model.clone();
        let mapping = build_model_mapping(entry)?;
        if models.insert(model.clone(), mapping).is_some() {
            return Err(format!("model_to_provider: duplicate model '{model}'").into());
        }
    }
    Ok(models)
}

/// Validate and construct one mapping entry.
fn build_model_mapping(entry: ModelProviderMappingConfig) -> Result<ModelProviderMapping, FilterError> {
    validate_mapping_value("model", &entry.model)?;
    validate_mapping_value("provider", &entry.provider)?;
    validate_mapping_value("target_model", &entry.target_model)?;
    let paths = validate_mapping_paths(&entry.model, entry.paths)?;
    let provider = HeaderValue::from_str(&entry.provider)
        .map_err(|error| FilterError::from(format!("model_to_provider: invalid provider value: {error}")))?;
    Ok(ModelProviderMapping {
        provider,
        target_model: entry.target_model,
        paths,
    })
}

/// Validate exact inference paths for one model mapping.
fn validate_mapping_paths(model: &str, paths: Vec<String>) -> Result<Vec<String>, FilterError> {
    if paths.is_empty() || paths.len() > MAX_PATHS_PER_MAPPING {
        return Err(format!("model_to_provider: each mapping must contain 1-{MAX_PATHS_PER_MAPPING} paths").into());
    }

    let mut validated = Vec::with_capacity(paths.len());
    for path in paths {
        if !path.starts_with('/') || path.contains(['?', '#']) || path.len() > 512 {
            return Err(
                format!("model_to_provider: path '{path}' must be an absolute path without query or fragment").into(),
            );
        }
        if validated.contains(&path) {
            return Err(format!("model_to_provider: duplicate path '{path}' for model '{model}'").into());
        }
        validated.push(path);
    }
    Ok(validated)
}

/// Default request body limit shared with JSON body filters.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Validate a public model, provider ID, or provider target model.
fn validate_mapping_value(field: &str, value: &str) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > MAX_MAPPING_VALUE_LEN || value.chars().any(char::is_control) {
        return Err(format!(
            "model_to_provider: {field} must be non-empty, control-free, and at most {MAX_MAPPING_VALUE_LEN} bytes"
        )
        .into());
    }
    Ok(())
}

#[async_trait]
impl HttpFilter for ModelToProviderFilter {
    fn name(&self) -> &'static str {
        "model_to_provider"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn needs_request_context(&self) -> bool {
        true
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one atomic model resolution decision owns header, metadata, and body mutation"
    )]
    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(mut value) = serde_json::from_slice::<Value>(raw) else {
            return Ok(FilterAction::Continue);
        };
        let Some(object) = value.as_object_mut() else {
            return Ok(FilterAction::Continue);
        };
        let Some(client_model) = object.get("model").and_then(Value::as_str) else {
            return Ok(FilterAction::Continue);
        };
        let Some(mapping) = self.models.get(client_model) else {
            return Ok(FilterAction::Continue);
        };
        let path = ctx.request.uri.path();
        if !mapping.paths.iter().any(|configured| configured == path) {
            return Ok(FilterAction::Continue);
        }

        ctx.request_headers_to_set
            .push((PROVIDER_HEADER.clone(), mapping.provider.clone()));
        ctx.set_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA, client_model);

        if object.get("model").and_then(Value::as_str) != Some(mapping.target_model.as_str()) {
            object.insert("model".to_owned(), Value::String(mapping.target_model.clone()));
            replace_json_body(body, &value, self.name(), "model").map_err(|error| {
                FilterError::from(format!("model_to_provider: request serialization failed: {error}"))
            })?;
        }

        Ok(FilterAction::Continue)
    }
}
