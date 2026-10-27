// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Principal-aware model catalog served from static configuration.

mod config;
mod visibility;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use async_trait::async_trait;
use bytes::{BufMut as _, Bytes, BytesMut};
use http::{HeaderMap, HeaderValue, Method, header};
use percent_encoding::percent_decode_str;
use praxis_filter::{
    AuthenticatedIdentity, FilterAction, FilterError, HttpFilter, HttpFilterContext, TerminalResponse,
};
use serde::Serialize;

use self::{
    config::{Capability, ModelCatalogConfig, ModelEntryConfig, validate_config},
    visibility::{PrincipalView, Visibility},
};

/// `owned_by` reported when an entry names neither `owned_by` nor `provider`.
const DEFAULT_OWNED_BY: &str = "praxis";

// -----------------------------------------------------------------------------
// ModelCatalogFilter
// -----------------------------------------------------------------------------

/// Serves an OpenAI-compatible `GET /v1/models` catalog filtered per principal.
///
/// The proxy answers list (`GET {list_path}`) and lookup
/// (`GET {list_path}/{id}`) requests itself from configured entries; nothing is
/// forwarded upstream. Each entry may restrict visibility to authenticated
/// subjects, roles, or teams. Hidden and unknown models both return the same
/// `404 model_not_found`, so callers cannot probe for models they cannot see.
/// Catalog metadata (provider, version, capabilities, cost tier) is
/// operator-declared and appears under a `praxis` key that OpenAI clients
/// ignore. Other paths pass through untouched.
pub struct ModelCatalogFilter {
    /// Path serving the model list.
    list_path: String,
    /// Compiled visibility rules, indexed by [`CatalogObject::visibility`].
    visibilities: Vec<Visibility>,
    /// Listed objects (entries and aliases) in config order.
    objects: Vec<CatalogObject>,
    /// Object index by model ID.
    by_id: HashMap<String, usize>,
}

/// One listed model object with its pre-serialized JSON.
struct CatalogObject {
    /// Index of the owning entry's visibility rule.
    visibility: usize,
    /// Serialized OpenAI model object.
    json: Bytes,
}

impl ModelCatalogFilter {
    /// Build the filter from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the config is malformed or invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ModelCatalogConfig = praxis_filter::parse_filter_config("model_catalog", config)?;
        validate_config(&cfg)?;
        Ok(Box::new(Self::build(cfg)?))
    }

    /// Compile validated config into visibility rules and serialized objects.
    fn build(cfg: ModelCatalogConfig) -> Result<Self, FilterError> {
        let mut visibilities = Vec::with_capacity(cfg.models.len());
        let mut objects = Vec::new();
        let mut by_id = HashMap::new();

        for mut entry in cfg.models {
            let visibility = visibilities.len();
            visibilities.push(Visibility::compile(entry.visible_to.take(), cfg.default_visibility));
            let payloads = entry_payloads(&entry, cfg.created)?;
            for (id, json) in std::iter::once(entry.id).chain(entry.aliases).zip(payloads) {
                by_id.insert(id, objects.len());
                objects.push(CatalogObject { visibility, json });
            }
        }

        Ok(Self {
            list_path: cfg.list_path,
            visibilities,
            objects,
            by_id,
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

    /// Whether `principal` may see `object`.
    fn visible(&self, object: &CatalogObject, principal: Option<PrincipalView<'_>>) -> bool {
        self.visibilities
            .get(object.visibility)
            .is_some_and(|rule| rule.allows(principal))
    }

    /// Assemble the list body from the objects visible to `principal`.
    fn list_body(&self, principal: Option<PrincipalView<'_>>) -> Bytes {
        const PREFIX: &[u8] = br#"{"object":"list","data":["#;
        const SUFFIX: &[u8] = b"]}";
        let visible: Vec<&CatalogObject> = self
            .objects
            .iter()
            .filter(|object| self.visible(object, principal))
            .collect();
        let len = PREFIX.len() + SUFFIX.len() + visible.iter().map(|o| o.json.len() + 1).sum::<usize>();
        let mut body = BytesMut::with_capacity(len);
        body.put_slice(PREFIX);
        for (index, object) in visible.iter().enumerate() {
            if index > 0 {
                body.put_u8(b',');
            }
            body.put_slice(&object.json);
        }
        body.put_slice(SUFFIX);
        body.freeze()
    }

    /// Respond to a lookup, hiding unknown and invisible models alike.
    fn retrieve(&self, raw_id: &str, principal: Option<PrincipalView<'_>>) -> TerminalResponse {
        let id = percent_decode_str(raw_id).decode_utf8_lossy();
        match self.by_id.get(id.as_ref()).and_then(|&index| self.objects.get(index)) {
            Some(object) if self.visible(object, principal) => json_response(200, object.json.clone()),
            _ => json_response(
                404,
                error_body(&format!("The model '{id}' does not exist"), "model_not_found"),
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
            let mut response = json_response(
                405,
                error_body("Method not allowed for the models endpoint", "method_not_allowed"),
            );
            response.headers.insert(header::ALLOW, HeaderValue::from_static("GET"));
            return Ok(FilterAction::TerminalResponse(Box::new(response)));
        }

        let principal = ctx.extensions.get::<AuthenticatedIdentity>().map(PrincipalView::from);
        let response = match route {
            CatalogRoute::List => json_response(200, self.list_body(principal)),
            CatalogRoute::Retrieve(id) => self.retrieve(id, principal),
        };
        Ok(FilterAction::TerminalResponse(Box::new(response)))
    }
}

// -----------------------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------------------

/// A request the catalog answers.
#[derive(Debug, PartialEq, Eq)]
enum CatalogRoute<'p> {
    /// `GET {list_path}`.
    List,
    /// `GET {list_path}/{id}` with the raw (still percent-encoded) ID.
    Retrieve(&'p str),
}

/// OpenAI model object with optional catalog metadata.
#[derive(Serialize)]
struct ModelObject<'a> {
    /// Model ID.
    id: &'a str,
    /// Always `model`.
    object: &'static str,
    /// Unix creation timestamp.
    created: u64,
    /// Owning organization.
    owned_by: &'a str,
    /// Catalog metadata, omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    praxis: Option<&'a CatalogMeta<'a>>,
}

/// Operator-declared catalog metadata under the `praxis` key.
#[derive(Clone, Copy, Serialize)]
struct CatalogMeta<'a> {
    /// Provider serving the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<&'a str>,
    /// Model version.
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<&'a str>,
    /// Declared API capabilities.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    capabilities: &'a [Capability],
    /// Coarse cost tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_tier: Option<&'a str>,
    /// Alias IDs of a canonical entry.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    aliases: &'a [String],
    /// Canonical ID of an alias object.
    #[serde(skip_serializing_if = "Option::is_none")]
    alias_of: Option<&'a str>,
}

impl CatalogMeta<'_> {
    /// Whether no metadata field would be serialized.
    fn is_empty(&self) -> bool {
        self.provider.is_none()
            && self.version.is_none()
            && self.capabilities.is_empty()
            && self.cost_tier.is_none()
            && self.aliases.is_empty()
            && self.alias_of.is_none()
    }
}

/// Serialize an entry's canonical object followed by one object per alias.
fn entry_payloads(entry: &ModelEntryConfig, default_created: u64) -> Result<Vec<Bytes>, FilterError> {
    let owned_by = entry
        .owned_by
        .as_deref()
        .or(entry.provider.as_deref())
        .unwrap_or(DEFAULT_OWNED_BY);
    let created = entry.created.unwrap_or(default_created);
    let meta = CatalogMeta {
        provider: entry.provider.as_deref(),
        version: entry.version.as_deref(),
        capabilities: &entry.capabilities,
        cost_tier: entry.cost_tier.as_deref(),
        aliases: &entry.aliases,
        alias_of: None,
    };

    let mut payloads = Vec::with_capacity(1 + entry.aliases.len());
    payloads.push(serialize_object(&entry.id, created, owned_by, &meta)?);
    let alias_meta = CatalogMeta {
        aliases: &[],
        alias_of: Some(&entry.id),
        ..meta
    };
    for alias in &entry.aliases {
        payloads.push(serialize_object(alias, created, owned_by, &alias_meta)?);
    }
    Ok(payloads)
}

/// Serialize one model object once, at build time.
fn serialize_object(id: &str, created: u64, owned_by: &str, meta: &CatalogMeta<'_>) -> Result<Bytes, FilterError> {
    let object = ModelObject {
        id,
        object: "model",
        created,
        owned_by,
        praxis: (!meta.is_empty()).then_some(meta),
    };
    serde_json::to_vec(&object)
        .map(Bytes::from)
        .map_err(|error| format!("model_catalog: failed to serialize model '{id}': {error}").into())
}

/// Build an OpenAI-style error body.
fn error_body(message: &str, code: &str) -> Bytes {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
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
