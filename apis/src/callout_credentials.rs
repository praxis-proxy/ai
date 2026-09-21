//! Per-user callout credentials captured at the trust boundary.
//!
//! The `CalloutCredentialsFilter` establishing filter reads
//! configured ingress headers, stores their values here as [`SecretString`] slots keyed by a
//! config-static slot id, and strips the ingress headers so they never reach an upstream.
//! Callout adapters read a slot through `stage_callout_identity`
//! and stage a per-user credential into the nested subrequest instead of a shared provider key.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_filter::{BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use secrecy::SecretString;
use serde::Deserialize;

/// Maximum number of credential slots one filter instance may declare.
const MAX_SLOTS: usize = 32;
/// Maximum byte length of a slot id.
const MAX_SLOT_ID_BYTES: usize = 128;

/// Routing/trust namespaces that must never be sourced from a client-supplied header.
const RESERVED_SOURCE_PREFIXES: &[&str] = &["x-praxis-", "x-mcp-", "x-ext-", "x-a2a-"];

/// YAML slot configuration before validation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSlot {
    /// Config-static slot identifier.
    id: String,
    /// Ingress header name to read the per-user secret from.
    header: String,
    /// Whether this slot's presence in the request is required.
    #[serde(default)]
    required: bool,
}

/// Top-level YAML configuration before validation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    /// List of credential slots to read and validate.
    slots: Vec<RawSlot>,
}

/// A validated per-user credential slot: config-static id + the ingress header it is read from.
#[derive(Debug, Clone)]
#[expect(dead_code, reason = "used in Task 3 runtime behavior")]
struct CredentialSlot {
    /// Config-static slot identifier.
    id: String,
    /// Validated ingress header name.
    header: HeaderName,
    /// Whether this slot's presence in the request is required.
    required: bool,
}

/// Establishing filter that captures per-user callout credentials from ingress headers.
#[derive(Debug)]
pub struct CalloutCredentialsFilter {
    /// Validated credential slots.
    #[expect(dead_code, reason = "used in Task 3 runtime behavior")]
    slots: Vec<CredentialSlot>,
}

impl CalloutCredentialsFilter {
    /// Parse and validate configuration, returning a boxed filter.
    ///
    /// # Errors
    /// Returns [`FilterError`] when the config has unknown fields, no slots, more than
    /// `MAX_SLOTS` slots, a duplicate slot id or source header, an oversized slot id, or a
    /// source header that is reserved, hop-by-hop, framing, or uses an internal-trust prefix.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let raw: RawConfig = parse_filter_config("callout_credentials", value)?;

        if raw.slots.is_empty() {
            return Err("callout_credentials: at least one slot is required".into());
        }
        if raw.slots.len() > MAX_SLOTS {
            return Err(format!("callout_credentials: too many slots (max {MAX_SLOTS})").into());
        }

        let slots = validate_slots(&raw.slots)?;
        Ok(Box::new(Self { slots }))
    }
}

#[async_trait]
impl HttpFilter for CalloutCredentialsFilter {
    fn name(&self) -> &'static str {
        "callout_credentials"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        _ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }
}

/// Validate raw slots into `CredentialSlot`s, rejecting duplicates and unsafe headers.
fn validate_slots(raw: &[RawSlot]) -> Result<Vec<CredentialSlot>, FilterError> {
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    let mut seen_headers: BTreeSet<String> = BTreeSet::new();
    let mut slots = Vec::with_capacity(raw.len());

    for slot in raw {
        if slot.id.is_empty() || slot.id.len() > MAX_SLOT_ID_BYTES {
            return Err(
                format!("callout_credentials: slot id must be 1..={MAX_SLOT_ID_BYTES} bytes")
                    .into(),
            );
        }
        if !seen_ids.insert(slot.id.as_str()) {
            return Err(
                format!("callout_credentials: duplicate slot id `{}`", slot.id).into(),
            );
        }
        let header = parse_source_header(&slot.header)?;
        if !seen_headers.insert(header.as_str().to_owned()) {
            return Err(
                format!("callout_credentials: duplicate source header `{}`", header.as_str())
                    .into(),
            );
        }
        slots.push(CredentialSlot {
            id: slot.id.clone(),
            header,
            required: slot.required,
        });
    }

    Ok(slots)
}

/// Validate a configured source header: parseable, not reserved/framing/hop-by-hop, not
/// routing-prefixed. Mirrors `state_owner_headers::parse_projection_header`.
fn parse_source_header(raw: &str) -> Result<HeaderName, FilterError> {
    if raw.is_empty() {
        return Err("callout_credentials: source header must not be empty".into());
    }
    let name = HeaderName::from_bytes(raw.as_bytes()).map_err(|error| -> FilterError {
        format!("callout_credentials: invalid header `{raw}`: {error}").into()
    })?;
    let lower = name.as_str();
    if lower == http::header::HOST.as_str() || lower == http::header::CONTENT_LENGTH.as_str() {
        return Err(format!("callout_credentials: source header `{lower}` is reserved").into());
    }
    if praxis_core::reserved_headers::HOP_BY_HOP_HEADERS.contains(&lower) {
        return Err(format!("callout_credentials: source header `{lower}` is hop-by-hop").into());
    }
    if praxis_core::reserved_headers::is_reserved(lower) {
        return Err(format!("callout_credentials: source header `{lower}` is reserved").into());
    }
    if RESERVED_SOURCE_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return Err(format!("callout_credentials: source header `{lower}` uses an internal-trust prefix").into());
    }
    Ok(name)
}

/// Request-scoped map of per-user callout credentials, keyed by config-static slot id.
///
/// Inserted into `RequestExtensions` by the `callout_credentials` filter. Slot ids are safe to
/// log (they come from static config); slot values are secret and never rendered.
#[derive(Default, Clone)]
pub struct CalloutCredentials {
    /// Secret values keyed by config-static slot id.
    slots: BTreeMap<String, SecretString>,
}

impl CalloutCredentials {
    /// Create an empty credential set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a secret under `slot`, replacing any prior value for that slot.
    pub fn insert(&mut self, slot: String, value: SecretString) {
        self.slots.insert(slot, value);
    }

    /// Look up the secret staged for `slot`, if any.
    pub fn get(&self, slot: &str) -> Option<&SecretString> {
        self.slots.get(slot)
    }

    /// True when no slots are populated.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Number of populated slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

impl std::fmt::Debug for CalloutCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalloutCredentials")
            .field("slots", &self.slots.keys().collect::<Vec<_>>())
            .field("values", &"[REDACTED]")
            .finish()
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[expect(clippy::unwrap_used, reason = "tests")]
    fn cfg(yaml: &str) -> Result<Box<dyn HttpFilter>, FilterError> {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        CalloutCredentialsFilter::from_config(&value)
    }

    #[test]
    fn accepts_valid_single_slot() {
        let f = cfg("slots:\n  - id: brave_search\n    header: x-user-brave-key\n    required: true\n");
        assert!(f.is_ok(), "valid config rejected: {:?}", f.err());
    }

    #[test]
    fn rejects_unknown_field() {
        let f = cfg("slots:\n  - id: a\n    header: x-user-a\nbogus: 1\n");
        assert!(f.is_err());
    }

    #[test]
    fn rejects_empty_slots() {
        assert!(cfg("slots: []\n").is_err());
    }

    #[test]
    fn rejects_duplicate_slot_id() {
        let f = cfg("slots:\n  - id: dup\n    header: x-user-a\n  - id: dup\n    header: x-user-b\n");
        assert!(f.is_err());
    }

    #[test]
    fn rejects_duplicate_source_header() {
        let f = cfg("slots:\n  - id: a\n    header: x-user-shared\n  - id: b\n    header: x-user-shared\n");
        assert!(f.is_err());
    }

    #[test]
    fn rejects_reserved_and_framing_source_headers() {
        assert!(cfg("slots:\n  - id: a\n    header: host\n").is_err());
        assert!(cfg("slots:\n  - id: a\n    header: content-length\n").is_err());
        assert!(cfg("slots:\n  - id: a\n    header: connection\n").is_err()); // hop-by-hop
    }

    #[test]
    fn rejects_routing_prefixed_source_headers() {
        assert!(cfg("slots:\n  - id: a\n    header: x-praxis-ai-model\n").is_err());
        assert!(cfg("slots:\n  - id: a\n    header: x-mcp-authorized\n").is_err());
        assert!(cfg("slots:\n  - id: a\n    header: x-ext-foo\n").is_err());
        assert!(cfg("slots:\n  - id: a\n    header: x-a2a-foo\n").is_err());
    }

    #[test]
    fn rejects_oversized_slot_id() {
        let big = "x".repeat(MAX_SLOT_ID_BYTES + 1);
        let f = cfg(&format!("slots:\n  - id: {big}\n    header: x-user-a\n"));
        assert!(f.is_err());
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;
    use secrecy::ExposeSecret as _;

    #[test]
    fn get_returns_inserted_secret() {
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("tok-123"));
        assert_eq!(creds.get("brave").unwrap().expose_secret(), "tok-123");
        assert!(creds.get("absent").is_none());
        assert!(!creds.is_empty());
        assert_eq!(creds.len(), 1);
    }

    #[test]
    fn debug_redacts_secret_values_but_lists_slot_names() {
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("super-secret"));
        let rendered = format!("{creds:?}");
        assert!(rendered.contains("brave"), "slot name should be visible: {rendered}");
        assert!(rendered.contains("REDACTED"), "must mark redaction: {rendered}");
        assert!(
            !rendered.contains("super-secret"),
            "secret value must never appear in Debug: {rendered}"
        );
    }

    #[test]
    fn empty_by_default() {
        let creds = CalloutCredentials::new();
        assert!(creds.is_empty());
        assert_eq!(creds.len(), 0);
    }
}
