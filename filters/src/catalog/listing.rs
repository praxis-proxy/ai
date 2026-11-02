// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Compiled catalog contents: configured overlay entries and the serialized
//! model objects built from them and from upstream model lists.

use std::collections::{HashMap, HashSet};

use bytes::{BufMut as _, Bytes, BytesMut};
use praxis_filter::FilterError;
use serde::{Deserialize, Serialize};

use super::{
    config::{DefaultVisibility, ModelEntryConfig},
    visibility::{PrincipalView, Visibility},
};

/// `owned_by` reported when the upstream names neither an owner nor a provider.
const DEFAULT_OWNED_BY: &str = "praxis";

/// `created` reported when the upstream does not report one.
const UNKNOWN_CREATED: u64 = 0;

// -----------------------------------------------------------------------------
// Overlay
// -----------------------------------------------------------------------------

/// Configured entries and their compiled visibility rules.
///
/// In static mode these entries are the whole catalog. In discovery mode they
/// only decorate the models upstreams report.
pub(super) struct Overlay {
    /// Entries with `visible_to` moved into [`Self::visibilities`].
    entries: Vec<ModelEntryConfig>,
    /// One rule per entry, then the default rule for unconfigured models.
    visibilities: Vec<Visibility>,
    /// Entry index by the ID upstreams report for it.
    by_upstream_id: HashMap<String, usize>,
    /// Every client-facing ID (entry IDs and aliases) the overlay claims.
    public_ids: HashSet<String>,
}

impl Overlay {
    /// Compile validated entries.
    pub(super) fn new(mut entries: Vec<ModelEntryConfig>, default: DefaultVisibility) -> Self {
        let mut visibilities = Vec::with_capacity(entries.len() + 1);
        let mut by_upstream_id = HashMap::with_capacity(entries.len());
        let mut public_ids = HashSet::new();
        for (index, entry) in entries.iter_mut().enumerate() {
            visibilities.push(Visibility::compile(entry.visible_to.take(), default));
            by_upstream_id.insert(entry.upstream_id().to_owned(), index);
            public_ids.insert(entry.id.clone());
            public_ids.extend(entry.aliases.iter().cloned());
        }
        visibilities.push(Visibility::compile(None, default));
        Self {
            entries,
            visibilities,
            by_upstream_id,
            public_ids,
        }
    }

    /// Index of the rule applied to upstream models without an entry.
    fn default_visibility(&self) -> usize {
        self.entries.len()
    }

    /// Whether `principal` may see `object`.
    pub(super) fn visible(&self, object: &CatalogObject, principal: Option<PrincipalView<'_>>) -> bool {
        self.visibilities
            .get(object.visibility)
            .is_some_and(|rule| rule.allows(principal))
    }
}

// -----------------------------------------------------------------------------
// Catalog
// -----------------------------------------------------------------------------

/// One model reported by an upstream `GET /v1/models`. Other fields are
/// ignored.
#[derive(Debug, Deserialize)]
pub(super) struct UpstreamModel {
    /// Upstream model ID.
    pub id: String,
    /// Unix creation timestamp, when reported.
    #[serde(default)]
    pub created: Option<u64>,
    /// Owning organization, when reported.
    #[serde(default)]
    pub owned_by: Option<String>,
}

/// One upstream's last known model list, for [`Catalog::from_upstreams`].
pub(super) struct UpstreamListing<'a> {
    /// Provider reported in catalog metadata for the listed models.
    pub provider: Option<&'a str>,
    /// Models the upstream reported.
    pub models: &'a [UpstreamModel],
}

/// Listed model objects, serialized once per build.
pub(super) struct Catalog {
    /// Listed objects (entries and aliases) in listing order.
    objects: Vec<CatalogObject>,
    /// Object index by model ID.
    by_id: HashMap<String, usize>,
}

/// One listed model object with its pre-serialized JSON.
pub(super) struct CatalogObject {
    /// Index of the object's rule in [`Overlay::visibilities`].
    visibility: usize,
    /// Serialized OpenAI model object.
    pub json: Bytes,
}

impl Catalog {
    /// List every configured entry, in config order.
    pub(super) fn from_entries(overlay: &Overlay) -> Result<Self, FilterError> {
        let mut catalog = Self::empty();
        for index in 0..overlay.entries.len() {
            catalog.push_entry(overlay, index, None, None)?;
        }
        Ok(catalog)
    }

    /// List the models upstreams report, in upstream order.
    ///
    /// A model with an entry takes that entry's ID, aliases, and visibility; entries no upstream reports are not
    /// listed. A model without an entry is listed as-is when `show_unconfigured` is set,
    /// unless an entry already claims its ID or an earlier upstream listed
    /// it.
    pub(super) fn from_upstreams(
        overlay: &Overlay,
        listings: &[UpstreamListing<'_>],
        show_unconfigured: bool,
    ) -> Result<Self, FilterError> {
        let mut catalog = Self::empty();
        let mut listed_entries = HashSet::new();
        for listing in listings {
            for model in listing.models {
                if let Some(&index) = overlay.by_upstream_id.get(&model.id) {
                    if listed_entries.insert(index) {
                        catalog.push_entry(overlay, index, Some(model), listing.provider)?;
                    }
                } else if show_unconfigured
                    && !overlay.public_ids.contains(&model.id)
                    && !catalog.by_id.contains_key(&model.id)
                {
                    catalog.push_unconfigured(overlay, model, listing.provider)?;
                }
            }
        }
        Ok(catalog)
    }

    /// An empty catalog.
    fn empty() -> Self {
        Self {
            objects: Vec::new(),
            by_id: HashMap::new(),
        }
    }

    /// Add an entry's canonical object and one object per alias.
    fn push_entry(
        &mut self,
        overlay: &Overlay,
        index: usize,
        upstream: Option<&UpstreamModel>,
        provider: Option<&str>,
    ) -> Result<(), FilterError> {
        let Some(entry) = overlay.entries.get(index) else {
            return Ok(());
        };
        let (owned_by, created) = upstream_identity(upstream, provider);
        let meta = CatalogMeta {
            provider,
            aliases: &entry.aliases,
            alias_of: None,
        };
        self.push(&entry.id, index, serialize_object(&entry.id, created, owned_by, &meta)?);

        let alias_meta = CatalogMeta {
            aliases: &[],
            alias_of: Some(&entry.id),
            ..meta
        };
        for alias in &entry.aliases {
            self.push(alias, index, serialize_object(alias, created, owned_by, &alias_meta)?);
        }
        Ok(())
    }

    /// Add an upstream model that has no entry.
    fn push_unconfigured(
        &mut self,
        overlay: &Overlay,
        model: &UpstreamModel,
        provider: Option<&str>,
    ) -> Result<(), FilterError> {
        let (owned_by, created) = upstream_identity(Some(model), provider);
        let meta = CatalogMeta {
            provider,
            ..CatalogMeta::EMPTY
        };
        let json = serialize_object(&model.id, created, owned_by, &meta)?;
        self.push(&model.id, overlay.default_visibility(), json);
        Ok(())
    }

    /// Record one serialized object.
    fn push(&mut self, id: &str, visibility: usize, json: Bytes) {
        self.by_id.insert(id.to_owned(), self.objects.len());
        self.objects.push(CatalogObject { visibility, json });
    }

    /// Look up a listed object by ID.
    pub(super) fn get(&self, id: &str) -> Option<&CatalogObject> {
        self.by_id.get(id).and_then(|&index| self.objects.get(index))
    }

    /// Assemble the list body from the objects visible to `principal`.
    pub(super) fn list_body(&self, overlay: &Overlay, principal: Option<PrincipalView<'_>>) -> Bytes {
        const PREFIX: &[u8] = br#"{"object":"list","data":["#;
        const SUFFIX: &[u8] = b"]}";
        let visible: Vec<&CatalogObject> = self
            .objects
            .iter()
            .filter(|object| overlay.visible(object, principal))
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
}

// -----------------------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------------------

/// Resolve `owned_by` and `created` from what the upstream reports, falling
/// back to its provider and then to defaults. Nothing is taken from config.
fn upstream_identity<'a>(upstream: Option<&'a UpstreamModel>, provider: Option<&'a str>) -> (&'a str, u64) {
    let owned_by = upstream
        .and_then(|model| model.owned_by.as_deref())
        .or(provider)
        .unwrap_or(DEFAULT_OWNED_BY);
    let created = upstream.and_then(|model| model.created).unwrap_or(UNKNOWN_CREATED);
    (owned_by, created)
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

/// Catalog metadata under the `praxis` key.
#[derive(Clone, Copy, Serialize)]
struct CatalogMeta<'a> {
    /// Provider of the upstream that lists the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<&'a str>,
    /// Alias IDs of a canonical entry.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    aliases: &'a [String],
    /// Canonical ID of an alias object.
    #[serde(skip_serializing_if = "Option::is_none")]
    alias_of: Option<&'a str>,
}

impl CatalogMeta<'_> {
    /// Metadata with no fields set.
    const EMPTY: CatalogMeta<'static> = CatalogMeta {
        provider: None,
        aliases: &[],
        alias_of: None,
    };

    /// Whether no metadata field would be serialized.
    fn is_empty(&self) -> bool {
        self.provider.is_none() && self.aliases.is_empty() && self.alias_of.is_none()
    }
}

/// Serialize one model object once, when the catalog is built.
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
