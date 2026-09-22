//! Per-user callout credentials captured at the trust boundary.
//!
//! The `CalloutCredentialsFilter` establishing filter reads
//! configured ingress headers, stores their values here as [`SecretString`] slots keyed by a
//! config-static slot id, and strips the ingress headers so they never reach an upstream.
//! Callout adapters read a slot through `stage_callout_identity`
//! and stage a per-user credential into the nested subrequest instead of a shared provider key.
//!
//! # YAML
//!
//! ```yaml
//! - filter: callout_credentials
//!   credentials:
//!     - slot: brave_search
//!       source_header: x-user-brave-key
//! ```

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_filter::{
    BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, TrustedHeaderMutation, parse_filter_config,
};
use secrecy::SecretString;
use serde::Deserialize;

use crate::state_owner::reject_owner;

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
    slot: String,
    /// Ingress header name to read the per-user secret from.
    source_header: String,
}

/// Top-level YAML configuration before validation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    /// Per-user credential slots to capture from ingress headers.
    credentials: Vec<RawSlot>,
}

/// A validated per-user credential slot: config-static id + the ingress header it is read from.
#[derive(Debug, Clone)]
struct CredentialSlot {
    /// Config-static slot identifier.
    id: String,
    /// Validated ingress header name.
    header: HeaderName,
}

/// Establishing filter that captures per-user callout credentials from ingress headers.
///
/// SECURITY: every configured `source_header` is trusted-boundary-owned input. The
/// authentication boundary that terminates ingress MUST unconditionally delete and
/// then set each credential source header on every request, so a client can never
/// spoof another user's credential by supplying the source header itself. This
/// filter strips the source headers before they reach any upstream, but it cannot
/// distinguish a boundary-set value from a client-supplied one; a deployment that
/// exposes a `source_header` a client can reach without that delete-then-set step
/// lets any caller inject an arbitrary per-user secret. Only route requests through
/// this filter behind a boundary that owns every configured source header.
#[derive(Debug)]
pub struct CalloutCredentialsFilter {
    /// Validated credential slots.
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

        if raw.credentials.is_empty() {
            return Err("callout_credentials: at least one credential is required".into());
        }
        if raw.credentials.len() > MAX_SLOTS {
            return Err(format!("callout_credentials: too many credentials (max {MAX_SLOTS})").into());
        }

        let slots = validate_slots(&raw.credentials)?;
        Ok(Box::new(Self { slots }))
    }

    /// Read all present slots from the effective header view, or reject on a duplicate.
    fn collect_present(
        &self,
        ctx: &HttpFilterContext<'_>,
        body_phase: bool,
    ) -> Result<Vec<(String, SecretString)>, FilterAction> {
        let view: Cow<'_, http::HeaderMap> = if body_phase {
            crate::callout_headers::effective_body_callout_headers(ctx, Cow::Borrowed(&ctx.request.headers))
        } else {
            Cow::Borrowed(&ctx.request.headers)
        };
        let mut present = Vec::new();
        for slot in &self.slots {
            match read_singular_slot(&view, &slot.header) {
                SlotValue::Single(value) => {
                    present.push((slot.id.clone(), SecretString::from(value)));
                },
                SlotValue::Missing => {},
                SlotValue::Duplicate => {
                    return Err(reject_owner(
                        400,
                        "duplicate_callout_credential",
                        &format!(
                            "callout credential header `{}` must appear exactly once",
                            slot.header.as_str()
                        ),
                    ));
                },
            }
        }
        Ok(present)
    }

    /// Strip every configured source header via the lifecycle-appropriate channel.
    fn queue_header_removal(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) {
        let ordered = body_phase && !ctx.pre_read_mutations.is_empty();
        for slot in &self.slots {
            ctx.request_headers_to_remove.push(slot.header.clone());
            if ordered {
                ctx.pre_read_mutations
                    .push(TrustedHeaderMutation::Remove(slot.header.clone()));
            }
        }
    }

    /// Establish per-user callout credentials and strip the ingress headers.
    fn resolve(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) -> FilterAction {
        if ctx.extensions.get::<CalloutCredentials>().is_some() {
            self.queue_header_removal(ctx, body_phase);
            return FilterAction::Continue;
        }
        let present = match self.collect_present(ctx, body_phase) {
            Ok(present) => present,
            Err(action) => return action,
        };
        if !present.is_empty() {
            let mut creds = CalloutCredentials::new();
            for (slot, value) in present {
                creds.insert(slot, value);
            }
            ctx.extensions.insert(creds);
        }
        self.queue_header_removal(ctx, body_phase);
        FilterAction::Continue
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

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(self.resolve(ctx, false))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        Ok(match self.resolve(ctx, true) {
            FilterAction::Continue => FilterAction::BodyDone,
            action => action,
        })
    }
}

/// Validate raw slots into `CredentialSlot`s, rejecting duplicates and unsafe headers.
fn validate_slots(raw: &[RawSlot]) -> Result<Vec<CredentialSlot>, FilterError> {
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    let mut seen_headers: BTreeSet<String> = BTreeSet::new();
    let mut slots = Vec::with_capacity(raw.len());

    for slot in raw {
        if slot.slot.is_empty() || slot.slot.len() > MAX_SLOT_ID_BYTES {
            return Err(format!("callout_credentials: slot must be 1..={MAX_SLOT_ID_BYTES} bytes").into());
        }
        if !seen_ids.insert(slot.slot.as_str()) {
            return Err(format!("callout_credentials: duplicate slot `{}`", slot.slot).into());
        }
        let header = parse_source_header(&slot.source_header)?;
        if !seen_headers.insert(header.as_str().to_owned()) {
            return Err(format!("callout_credentials: duplicate source header `{}`", header.as_str()).into());
        }
        slots.push(CredentialSlot {
            id: slot.slot.clone(),
            header,
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
    let name = HeaderName::from_bytes(raw.as_bytes())
        .map_err(|error| -> FilterError { format!("callout_credentials: invalid header `{raw}`: {error}").into() })?;
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

/// Singular-value outcome for one configured slot header.
enum SlotValue {
    /// Exactly one non-empty UTF-8 value.
    Single(String),
    /// Absent, empty, or non-UTF-8 — treated as unpopulated (no reject).
    Missing,
    /// More than one value — a client error.
    Duplicate,
}

/// Read one slot header with singular-value semantics from an already-effective view.
fn read_singular_slot(headers: &http::HeaderMap, header: &HeaderName) -> SlotValue {
    let mut it = headers.get_all(header).iter();
    match (it.next(), it.next()) {
        (None, _) => SlotValue::Missing,
        (Some(_), Some(_)) => SlotValue::Duplicate,
        (Some(v), None) => match v.to_str() {
            Ok(s) if !s.is_empty() => SlotValue::Single(s.to_owned()),
            _ => SlotValue::Missing,
        },
    }
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
        let f = cfg("credentials:\n  - slot: brave_search\n    source_header: x-user-brave-key\n");
        assert!(f.is_ok(), "valid config rejected: {:?}", f.err());
    }

    #[test]
    fn rejects_unknown_field() {
        let f = cfg("credentials:\n  - slot: a\n    source_header: x-user-a\nbogus: 1\n");
        assert!(f.is_err(), "an unknown top-level field must be rejected");
    }

    #[test]
    fn rejects_empty_slots() {
        assert!(
            cfg("credentials: []\n").is_err(),
            "at least one credential slot is required"
        );
    }

    #[test]
    fn rejects_duplicate_slot_id() {
        let f = cfg(
            "credentials:\n  - slot: dup\n    source_header: x-user-a\n  - slot: dup\n    source_header: x-user-b\n",
        );
        assert!(f.is_err(), "a duplicate slot id must be rejected");
    }

    #[test]
    fn rejects_duplicate_source_header() {
        let f = cfg(
            "credentials:\n  - slot: a\n    source_header: x-user-shared\n  - slot: b\n    source_header: x-user-shared\n",
        );
        assert!(f.is_err(), "a source header shared across two slots must be rejected");
    }

    #[test]
    fn rejects_reserved_and_framing_source_headers() {
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: host\n").is_err(),
            "`host` is a framing header and must be rejected"
        );
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: content-length\n").is_err(),
            "`content-length` is a framing header and must be rejected"
        );
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: connection\n").is_err(),
            "`connection` is a hop-by-hop header and must be rejected"
        );
    }

    #[test]
    fn rejects_routing_prefixed_source_headers() {
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: x-praxis-ai-model\n").is_err(),
            "the internal-trust `x-praxis-` prefix must be rejected"
        );
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: x-mcp-authorized\n").is_err(),
            "the reserved `x-mcp-` prefix must be rejected"
        );
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: x-ext-foo\n").is_err(),
            "the reserved `x-ext-` prefix must be rejected"
        );
        assert!(
            cfg("credentials:\n  - slot: a\n    source_header: x-a2a-foo\n").is_err(),
            "the reserved `x-a2a-` prefix must be rejected"
        );
    }

    #[test]
    fn rejects_oversized_slot_id() {
        let big = "x".repeat(MAX_SLOT_ID_BYTES + 1);
        let f = cfg(&format!("credentials:\n  - slot: {big}\n    source_header: x-user-a\n"));
        assert!(f.is_err(), "a slot id larger than MAX_SLOT_ID_BYTES must be rejected");
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use secrecy::ExposeSecret as _;

    use super::*;

    #[test]
    fn get_returns_inserted_secret() {
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("tok-123"));
        assert_eq!(creds.get("brave").unwrap().expose_secret(), "tok-123");
        assert!(creds.get("absent").is_none(), "an unset slot resolves to None");
        assert!(!creds.is_empty(), "a populated store is not empty");
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
        assert!(creds.is_empty(), "a fresh store is empty");
        assert_eq!(creds.len(), 0);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod runtime_tests {
    use http::{HeaderValue, Method};
    use secrecy::ExposeSecret as _;

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    /// Make a filter with one slot for testing.
    fn make_filter() -> CalloutCredentialsFilter {
        CalloutCredentialsFilter {
            slots: vec![CredentialSlot {
                id: "brave".to_owned(),
                header: HeaderName::from_static("x-user-brave-key"),
            }],
        }
    }

    #[tokio::test]
    async fn installs_slot_and_strips_ingress_header() {
        let filter = make_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert("x-user-brave-key", HeaderValue::from_static("tok-abc"));
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "installing a slot must continue"
        );

        let creds = ctx.extensions.get::<CalloutCredentials>().unwrap();
        assert_eq!(creds.get("brave").unwrap().expose_secret(), "tok-abc");
        assert!(
            ctx.request_headers_to_remove.iter().any(|n| n == "x-user-brave-key"),
            "the ingress source header must be queued for removal"
        );
    }

    #[tokio::test]
    async fn missing_optional_header_leaves_slot_unpopulated_without_reject() {
        let filter = make_filter();
        let request = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "a missing optional header must continue"
        );

        assert!(
            ctx.extensions
                .get::<CalloutCredentials>()
                .is_none_or(|c| c.get("brave").is_none()),
            "an absent source header leaves the slot unpopulated"
        );
        assert!(
            ctx.request_headers_to_remove.iter().any(|n| n == "x-user-brave-key"),
            "the source header is still stripped even when absent-valued"
        );
    }

    #[tokio::test]
    #[expect(clippy::panic, reason = "tests")]
    async fn duplicate_header_value_rejects_400() {
        let filter = make_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert("x-user-brave-key", HeaderValue::from_static("tok-1"));
        request
            .headers
            .append("x-user-brave-key", HeaderValue::from_static("tok-2"));
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("expected rejection");
        };
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(
            body.pointer("/error/type").and_then(|v| v.as_str()),
            Some("invalid_request_error")
        );
        assert_eq!(
            body.pointer("/error/code").and_then(|v| v.as_str()),
            Some("duplicate_callout_credential")
        );
    }

    #[tokio::test]
    async fn idempotent_when_already_installed() {
        let filter = make_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert("x-user-brave-key", HeaderValue::from_static("new"));
        let mut ctx = make_filter_context(&request);
        let mut prior_creds = CalloutCredentials::new();
        prior_creds.insert("brave".to_owned(), SecretString::from("prior"));
        ctx.extensions.insert(prior_creds);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "an already-installed store must continue"
        );

        let creds = ctx.extensions.get::<CalloutCredentials>().unwrap();
        assert_eq!(creds.get("brave").unwrap().expose_secret(), "prior");
        assert!(
            ctx.request_headers_to_remove.iter().any(|n| n == "x-user-brave-key"),
            "the source header is stripped even when the store is already installed"
        );
    }

    #[tokio::test]
    async fn body_phase_maps_continue_to_body_done() {
        let filter = make_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert("x-user-brave-key", HeaderValue::from_static("tok-xyz"));
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::BodyDone),
            "the body phase maps Continue to BodyDone"
        );

        let creds = ctx.extensions.get::<CalloutCredentials>().unwrap();
        assert_eq!(creds.get("brave").unwrap().expose_secret(), "tok-xyz");
    }
}
