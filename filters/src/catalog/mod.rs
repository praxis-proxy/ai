// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Principal-aware model catalog, discovered from upstreams or configured
//! statically.

mod config;
mod discovery;
mod listing;
mod visibility;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method, header};
use percent_encoding::percent_decode_str;
use praxis_ai_apis::subrequest::SubRequestClient;
use praxis_filter::{
    AuthenticatedIdentity, FilterAction, FilterError, HttpFilter, HttpFilterContext, TerminalResponse,
};

use self::{
    config::{ModelCatalogConfig, validate_config},
    discovery::Discovery,
    listing::{Catalog, Overlay},
    visibility::PrincipalView,
};

// -----------------------------------------------------------------------------
// ModelCatalogFilter
// -----------------------------------------------------------------------------

/// Serves an OpenAI-compatible `GET /v1/models` catalog filtered per principal.
///
/// The proxy answers list (`GET {list_path}`) and lookup
/// (`GET {list_path}/{id}`) requests itself. With `upstreams` configured, the
/// models those upstreams report decide what exists: their `GET /v1/models`
/// lists are fetched on first use and refreshed every `refresh_interval_secs`
/// in the background, and an upstream that fails keeps its last known list.
/// Configured `models` entries then only overlay client-facing IDs, aliases,
/// metadata, and visibility; an entry no upstream reports is not listed.
/// Without `upstreams`, the configured entries are the catalog. Each entry may
/// restrict visibility to authenticated subjects, roles, or teams. Hidden and
/// unknown models both return the same `404 model_not_found`, so callers
/// cannot probe for models they cannot see. Catalog metadata (provider,
/// version, capabilities, cost tier) is operator-declared and appears under a
/// `praxis` key that OpenAI clients ignore. Other paths pass through
/// untouched.
pub struct ModelCatalogFilter {
    /// Path serving the model list.
    list_path: String,
    /// Configured entries and visibility rules.
    overlay: Arc<Overlay>,
    /// Where listed models come from.
    source: CatalogSource,
}

/// Where the catalog's models come from.
enum CatalogSource {
    /// Configured entries, built once.
    Static(Catalog),
    /// Upstream model lists, refreshed periodically.
    Discovered(Arc<Discovery>),
}

impl ModelCatalogFilter {
    /// Build the filter from YAML config with an isolated sub-request client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the config is malformed or invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Self::from_config_with_client(config, crate::isolated_subrequest_client(4))
    }

    /// Build the filter from YAML config using a shared sub-request client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the config is malformed or invalid.
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ModelCatalogConfig = praxis_filter::parse_filter_config("model_catalog", config)?;
        validate_config(&cfg)?;
        Ok(Box::new(Self::build(cfg, client)?))
    }

    /// Compile validated config into an overlay and catalog source.
    fn build(mut cfg: ModelCatalogConfig, client: SubRequestClient) -> Result<Self, FilterError> {
        let overlay = Arc::new(Overlay::new(
            std::mem::take(&mut cfg.models),
            cfg.default_visibility,
            cfg.created,
        ));
        let source = if cfg.upstreams.is_empty() {
            CatalogSource::Static(Catalog::from_entries(&overlay)?)
        } else {
            CatalogSource::Discovered(Arc::new(Discovery::new(&mut cfg, Arc::clone(&overlay), client)?))
        };
        Ok(Self {
            list_path: cfg.list_path,
            overlay,
            source,
        })
    }

    /// Classify a request path against the catalog routes.
    fn route<'p>(&self, path: &'p str) -> Option<CatalogRoute<'p>> {
        let path = path.strip_suffix('/').filter(|p| !p.is_empty()).unwrap_or(path);
        let rest = path.strip_prefix(self.list_path.as_str())?;
        if rest.is_empty() {
            return Some(CatalogRoute::List);
        }
        rest.strip_prefix('/').map(CatalogRoute::Retrieve)
    }

    /// Answer a catalog route from `catalog`.
    fn respond(
        &self,
        catalog: &Catalog,
        route: CatalogRoute<'_>,
        principal: Option<PrincipalView<'_>>,
    ) -> TerminalResponse {
        match route {
            CatalogRoute::List => json_response(200, catalog.list_body(&self.overlay, principal)),
            CatalogRoute::Retrieve(raw_id) => self.retrieve(catalog, raw_id, principal),
        }
    }

    /// Respond to a lookup, hiding unknown and invisible models alike.
    fn retrieve(&self, catalog: &Catalog, raw_id: &str, principal: Option<PrincipalView<'_>>) -> TerminalResponse {
        let id = percent_decode_str(raw_id).decode_utf8_lossy();
        match catalog.get(id.as_ref()) {
            Some(object) if self.overlay.visible(object, principal) => json_response(200, object.json.clone()),
            _ => json_response(
                404,
                error_body(
                    &format!("The model '{id}' does not exist"),
                    "invalid_request_error",
                    "model_not_found",
                ),
            ),
        }
    }
}

#[async_trait]
impl HttpFilter for ModelCatalogFilter {
    fn name(&self) -> &'static str {
        "model_catalog"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(route) = self.route(ctx.request.uri.path()) else {
            return Ok(FilterAction::Continue);
        };
        if ctx.request.method != Method::GET {
            return Ok(FilterAction::TerminalResponse(Box::new(method_not_allowed())));
        }

        let response = match &self.source {
            CatalogSource::Static(catalog) => {
                let principal = ctx.extensions.get::<AuthenticatedIdentity>().map(PrincipalView::from);
                self.respond(catalog, route, principal)
            },
            CatalogSource::Discovered(discovery) => {
                let snapshot = discovery.snapshot().await;
                let principal = ctx.extensions.get::<AuthenticatedIdentity>().map(PrincipalView::from);
                match snapshot.as_ref().and_then(|snapshot| snapshot.catalog.as_deref()) {
                    Some(catalog) => self.respond(catalog, route, principal),
                    None => catalog_unavailable(),
                }
            },
        };
        Ok(FilterAction::TerminalResponse(Box::new(response)))
    }
}

// -----------------------------------------------------------------------------
// Responses
// -----------------------------------------------------------------------------

/// A request the catalog answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogRoute<'p> {
    /// `GET {list_path}`.
    List,
    /// `GET {list_path}/{id}` with the raw (still percent-encoded) ID.
    Retrieve(&'p str),
}

/// `405` for any method other than `GET` on a catalog path.
fn method_not_allowed() -> TerminalResponse {
    let mut response = json_response(
        405,
        error_body(
            "Method not allowed for the models endpoint",
            "invalid_request_error",
            "method_not_allowed",
        ),
    );
    response.headers.insert(header::ALLOW, HeaderValue::from_static("GET"));
    response
}

/// `503` while no upstream model list has ever been retrieved.
fn catalog_unavailable() -> TerminalResponse {
    json_response(
        503,
        error_body(
            "The model catalog is unavailable: no upstream model list has been retrieved",
            "server_error",
            "model_catalog_unavailable",
        ),
    )
}

/// Build an OpenAI-style error body.
fn error_body(message: &str, kind: &str, code: &str) -> Bytes {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": kind,
            "param": null,
            "code": code,
        }
    });
    Bytes::from(body.to_string())
}

/// Build a JSON terminal response that must not be cached across principals.
fn json_response(status: u16, body: Bytes) -> TerminalResponse {
    let mut headers = HeaderMap::with_capacity(2);
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("private, no-store"));
    TerminalResponse::new(status).with_headers(headers).with_body(body)
}
