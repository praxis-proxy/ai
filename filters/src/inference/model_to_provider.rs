// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Resolves stable client-facing model IDs to existing upstream clusters.
//!
//! An exact model/path match selects a configured cluster, rewrites the request
//! body's `model`, and preserves the public model ID for metering and response
//! adapters. Unmatched bodies remain unchanged; `fallback_upstream` optionally
//! selects their cluster. The load balancer owns endpoint selection and retries.
//! Configuration is local to the filter pipeline; this filter does not watch
//! Kubernetes resources or perform request-time control-plane lookups.

use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::{MODEL_PROVIDER_CLIENT_MODEL_METADATA, json_body::replace_json_body};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes, parse_filter_config,
};
use serde::Deserialize;
use serde_json::Value;

mod model;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum mappings accepted in one filter config.
const MAX_MODEL_MAPPINGS: usize = 1024;
/// Maximum paths accepted for one model mapping.
const MAX_PATHS_PER_MAPPING: usize = 16;
/// Maximum byte length for model and upstream identifiers.
const MAX_MAPPING_VALUE_LEN: usize = 253;

/// Configures a client model's upstream cluster and target model identity.
///
/// ```yaml
/// filter: model_to_provider
/// models:
///   - model: claude-sonnet-4-5
///     upstream: vertex-prod
///     target_model: claude-sonnet-4-5@20250929
///     paths: [/v1/messages, /v1/messages/count_tokens]
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelToProviderConfig {
    /// Exact client-facing model mappings.
    models: Vec<ModelProviderMappingConfig>,
    /// Existing load-balancer cluster selected when no model/path mapping matches.
    #[serde(default)]
    fallback_upstream: Option<String>,
    /// Maximum JSON request body buffered to inspect the model field.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// One public-model to upstream mapping.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelProviderMappingConfig {
    /// Exact model ID sent by the client.
    model: String,
    /// Existing load-balancer cluster selected for this model.
    upstream: String,
    /// Provider-specific model written into the request body.
    target_model: String,
    /// Exact inference paths on which this mapping applies.
    paths: Vec<String>,
}

/// Validated mapping used on the request path.
#[derive(Debug)]
struct ModelProviderMapping {
    /// Shared configured cluster name selected during the request phase.
    upstream: Arc<str>,
    /// Provider-specific request-body model.
    target_model: String,
    /// Exact request paths accepted for this client model.
    paths: Vec<String>,
}

/// Maps stable client-facing model IDs to existing upstream clusters and
/// provider-specific target models.
///
/// An exact model/path match rewrites the JSON body model and records the
/// client ID for metering and provider response adapters. During the request
/// phase, the filter selects the mapping's `upstream` cluster. A following
/// `load_balancer` owns endpoint selection, health checks, and retries; no
/// header-matching `router` or provider registry is needed.
///
/// Unknown models and paths keep their bodies unchanged and select
/// `fallback_upstream` when configured. Without it, this filter leaves cluster
/// selection unchanged; a load balancer still needs a selected cluster to
/// forward the request. Do not combine this selector with a competing router
/// before the same load balancer.
///
/// The normal server pipeline validation checks mapped and fallback cluster
/// names against the load balancers on startup and reload. Cluster application
/// metadata is optional for routing. Provider adapters can read
/// `selected_application_provider()` after load balancing and must separately
/// declare any application metadata they require on those clusters.
///
/// The filter does not restore the client-facing model ID in responses, so a
/// provider that echoes the model it served returns `target_model` to the
/// client unless a provider response adapter reads the client-facing ID back
/// from the `model_to_provider.client_model` metadata key.
///
/// # YAML configuration
///
/// ```yaml
/// filter: model_to_provider
/// models:
///   - model: claude-sonnet-4-5
///     upstream: vertex-prod
///     target_model: claude-sonnet-4-5@20250929
///     paths: [/v1/messages, /v1/messages/count_tokens]
/// ```
pub struct ModelToProviderFilter {
    /// Exact public model ID to upstream mapping table.
    models: HashMap<String, ModelProviderMapping>,
    /// Configured paths indexed once to bypass unrelated request bodies.
    paths: HashSet<String>,
    /// Optional cluster for requests that match no mapping.
    fallback_upstream: Option<Arc<str>>,
    /// Maximum buffered request body size.
    max_body_bytes: usize,
}

/// Per-filter selection staged during body pre-read for the request phase.
struct ResolvedUpstream(Arc<str>);

impl ModelToProviderFilter {
    /// Construct from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, duplicate, or invalid mappings and for an
    /// invalid request-body limit.
    ///
    /// # Example
    ///
    /// ```
    /// use praxis_ai_filters::ModelToProviderFilter;
    ///
    /// let config = serde_yaml::from_str(
    ///     "models:\n  - model: public-model\n    paths: [/v1/messages]\n    upstream: local\n    target_model: backend-model",
    /// ).unwrap();
    /// let filter = ModelToProviderFilter::from_config(&config).unwrap();
    /// assert_eq!(filter.name(), "model_to_provider");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: ModelToProviderConfig = parse_filter_config("model_to_provider", config)?;
        if config.models.is_empty() || config.models.len() > MAX_MODEL_MAPPINGS {
            return Err(format!("model_to_provider: models must contain 1-{MAX_MODEL_MAPPINGS} mappings").into());
        }
        validate_max_body_bytes("model_to_provider", config.max_body_bytes)?;
        if let Some(upstream) = &config.fallback_upstream {
            validate_mapping_value("fallback_upstream", upstream)?;
        }
        let models = build_model_mappings(config.models)?;
        let paths = models
            .values()
            .flat_map(|mapping| mapping.paths.iter().cloned())
            .collect();
        Ok(Box::new(Self {
            models,
            paths,
            fallback_upstream: config.fallback_upstream.map(Arc::from),
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
        let (model, mapping) = build_model_mapping(entry)?;
        match models.entry(model) {
            Entry::Vacant(slot) => {
                slot.insert(mapping);
            },
            Entry::Occupied(slot) => {
                return Err(format!("model_to_provider: duplicate model '{}'", slot.key()).into());
            },
        }
    }
    Ok(models)
}

/// Validate and construct one mapping entry.
fn build_model_mapping(entry: ModelProviderMappingConfig) -> Result<(String, ModelProviderMapping), FilterError> {
    validate_mapping_value("model", &entry.model)?;
    validate_mapping_value("upstream", &entry.upstream)?;
    validate_mapping_value("target_model", &entry.target_model)?;
    let paths = validate_mapping_paths(&entry.model, entry.paths)?;
    Ok((
        entry.model,
        ModelProviderMapping {
            upstream: Arc::from(entry.upstream),
            target_model: entry.target_model,
            paths,
        },
    ))
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

/// Validate a public model, cluster name, or provider target model.
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

    fn selects_cluster(&self) -> bool {
        true
    }

    fn selected_clusters(&self) -> Vec<String> {
        let mut clusters: Vec<String> = self
            .models
            .values()
            .map(|mapping| mapping.upstream.to_string())
            .chain(self.fallback_upstream.iter().map(ToString::to_string))
            .collect();
        clusters.sort_unstable();
        clusters.dedup();
        clusters
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

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let upstream = ctx
            .get_filter_state::<ResolvedUpstream>()
            .map(|resolved| Arc::clone(&resolved.0))
            .or_else(|| self.fallback_upstream.clone());
        if let Some(upstream) = upstream {
            ctx.cluster = Some(upstream);
            ctx.route_retry_policy = None;
            ctx.hedge_policy = None;
            ctx.metrics_route = None;
        }
        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one atomic model resolution decision owns selection, metadata, and body mutation"
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
        ctx.remove_filter_state::<ResolvedUpstream>();
        ctx.filter_metadata.remove(MODEL_PROVIDER_CLIENT_MODEL_METADATA);

        let path = ctx.request.uri.path();
        if !self.paths.contains(path) {
            return Ok(FilterAction::Continue);
        }
        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };
        let Some(client_model) = model::peek(raw) else {
            return Ok(FilterAction::Continue);
        };
        let Some(mapping) = self.models.get(client_model.as_ref()) else {
            return Ok(FilterAction::Continue);
        };
        if !mapping.paths.iter().any(|configured| configured == path) {
            return Ok(FilterAction::Continue);
        }
        // Validate matched requests with Value before publishing any selection:
        // skipped fields may contain out-of-range numbers or invalid UTF-8.
        let Ok(mut value) = serde_json::from_slice::<Value>(raw) else {
            return Ok(FilterAction::Continue);
        };
        let Some(object) = value.as_object_mut() else {
            return Ok(FilterAction::Continue);
        };
        let Some(client_model) = object.get("model").and_then(Value::as_str) else {
            return Ok(FilterAction::Continue);
        };
        let needs_rewrite = client_model != mapping.target_model;
        ctx.insert_filter_state(ResolvedUpstream(Arc::clone(&mapping.upstream)));
        ctx.set_metadata(MODEL_PROVIDER_CLIENT_MODEL_METADATA, client_model);

        if needs_rewrite {
            object.insert("model".to_owned(), Value::String(mapping.target_model.clone()));
            replace_json_body(body, &value, self.name(), "model").map_err(|error| {
                FilterError::from(format!("model_to_provider: request serialization failed: {error}"))
            })?;
        }

        Ok(FilterAction::Continue)
    }
}

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
