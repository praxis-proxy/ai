//! Per-user callout secrets captured at the trust boundary.
//!
//! The `CalloutCredentialsFilter` establishing filter reads
//! configured ingress headers, stores their values here as typed [`SecretString`] slots keyed by a
//! config-static slot id, and strips the ingress headers so they never reach an upstream. Callout
//! adapters read credential slots through `stage_callout_identity`; MCP additionally reads an
//! assertion slot and injects it only into configured connector requests.
//!
//! # YAML
//!
//! ```yaml
//! - filter: callout_credentials
//!   credentials:
//!     - slot: brave_search
//!       source_header: x-user-brave-key
//!   assertions:
//!     - slot: mcp_gateway
//!       source_header: x-mcp-authorized
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

/// Maximum number of secret slots one filter instance may declare.
const MAX_SLOTS: usize = 32;
/// Maximum byte length of a slot id.
const MAX_SLOT_ID_BYTES: usize = 128;
/// Maximum authorization assertion size accepted from the trusted boundary.
const MAX_ASSERTION_BYTES: usize = 4_096;

/// Fixed trusted-boundary MCP assertion header.
pub(crate) const MCP_AUTHORIZED_HEADER: HeaderName = HeaderName::from_static("x-mcp-authorized");

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
    #[serde(default)]
    credentials: Vec<RawSlot>,
    /// Opaque authorization-assertion slots to capture from trusted ingress headers.
    #[serde(default)]
    assertions: Vec<RawSlot>,
}

/// A validated per-user credential slot: config-static id + the ingress header it is read from.
#[derive(Debug, Clone)]
struct CredentialSlot {
    /// Config-static slot identifier.
    id: String,
    /// Validated ingress header name.
    header: HeaderName,
}

/// Establishing filter that captures typed per-user callout secrets from ingress headers.
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
///
/// Place this filter once in the outer request chain, before body pre-read callouts
/// and any `iterative_request_router`. It establishes one
/// [`CalloutCredentials`] map that the router carries across iterations; callout
/// consumers inside or outside the router select their own slot. Do not repeat the
/// establishing filter inside router steps.
#[derive(Debug)]
pub struct CalloutCredentialsFilter {
    /// Validated credential slots.
    credential_slots: Vec<CredentialSlot>,
    /// Validated authorization-assertion slots.
    assertion_slots: Vec<CredentialSlot>,
}

impl CalloutCredentialsFilter {
    /// Parse and validate configuration, returning a boxed filter.
    ///
    /// # Errors
    /// Returns [`FilterError`] when the config has unknown fields, no slots, more than
    /// `MAX_SLOTS` slots, a duplicate slot id or source header, an oversized slot id, or an
    /// unsafe source header. Assertion sources are deliberately restricted to
    /// `x-mcp-authorized`.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let raw: RawConfig = parse_filter_config("callout_credentials", value)?;

        let slot_count = raw.credentials.len().saturating_add(raw.assertions.len());
        if slot_count == 0 {
            return Err("callout_credentials: at least one credential or assertion is required".into());
        }
        if slot_count > MAX_SLOTS {
            return Err(format!("callout_credentials: too many secret slots (max {MAX_SLOTS})").into());
        }

        let credential_slots = validate_slots(&raw.credentials, SlotKind::Credential)?;
        let assertion_slots = validate_slots(&raw.assertions, SlotKind::Assertion)?;
        reject_duplicate_source_headers(&credential_slots, &assertion_slots)?;
        Ok(Box::new(Self {
            credential_slots,
            assertion_slots,
        }))
    }

    /// Build the request-scoped secret maps from the effective header view.
    fn collect_secrets(
        &self,
        ctx: &HttpFilterContext<'_>,
        body_phase: bool,
    ) -> Result<CalloutCredentials, FilterAction> {
        let view: Cow<'_, http::HeaderMap> = if body_phase {
            crate::callout_headers::effective_body_callout_headers(ctx, Cow::Borrowed(&ctx.request.headers))
        } else {
            Cow::Borrowed(&ctx.request.headers)
        };
        let mut credentials = CalloutCredentials::new();
        self.collect_credential_slots(&view, &mut credentials)?;
        self.collect_assertion_slots(&view, &mut credentials)?;
        Ok(credentials)
    }

    /// Capture destination credentials without exposing assertion slots to generic consumers.
    fn collect_credential_slots(
        &self,
        view: &http::HeaderMap,
        credentials: &mut CalloutCredentials,
    ) -> Result<(), FilterAction> {
        for slot in &self.credential_slots {
            match read_singular_slot(view, &slot.header) {
                SlotValue::Single(value) => {
                    credentials.insert(slot.id.clone(), SecretString::from(value));
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
        Ok(())
    }

    /// Capture connector assertions into their dedicated typed map.
    fn collect_assertion_slots(
        &self,
        view: &http::HeaderMap,
        credentials: &mut CalloutCredentials,
    ) -> Result<(), FilterAction> {
        for slot in &self.assertion_slots {
            match read_singular_slot(view, &slot.header) {
                SlotValue::Single(value) if value.len() <= MAX_ASSERTION_BYTES => {
                    credentials.insert_assertion(slot.id.clone(), SecretString::from(value));
                },
                SlotValue::Single(_) => {
                    return Err(reject_owner(
                        400,
                        "callout_authorization_too_large",
                        "x-mcp-authorized exceeds the 4096-byte limit",
                    ));
                },
                SlotValue::Missing => {},
                SlotValue::Duplicate => {
                    return Err(reject_owner(
                        400,
                        "duplicate_callout_authorization",
                        "x-mcp-authorized must appear exactly once",
                    ));
                },
            }
        }
        Ok(())
    }

    /// Strip every configured source header via the lifecycle-appropriate channel.
    fn queue_header_removal(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) {
        let ordered = body_phase && !ctx.pre_read_mutations.is_empty();
        for slot in self.credential_slots.iter().chain(&self.assertion_slots) {
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
        let credentials = match self.collect_secrets(ctx, body_phase) {
            Ok(credentials) => credentials,
            Err(action) => return action,
        };
        if !credentials.is_empty() {
            ctx.extensions.insert(credentials);
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

/// The security meaning of a configured secret slot.
#[derive(Clone, Copy)]
enum SlotKind {
    /// A destination-bound callout credential.
    Credential,
    /// An opaque MCP Gateway authorization assertion.
    Assertion,
}

/// Validate raw slots into `CredentialSlot`s, rejecting duplicates and unsafe headers.
fn validate_slots(raw: &[RawSlot], kind: SlotKind) -> Result<Vec<CredentialSlot>, FilterError> {
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
        let header = match kind {
            SlotKind::Credential => parse_credential_source_header(&slot.source_header)?,
            SlotKind::Assertion => parse_assertion_source_header(&slot.source_header)?,
        };
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

/// Reject one ingress source assigned to more than one secret kind.
fn reject_duplicate_source_headers(
    credential_slots: &[CredentialSlot],
    assertion_slots: &[CredentialSlot],
) -> Result<(), FilterError> {
    let credential_headers: BTreeSet<&str> = credential_slots.iter().map(|slot| slot.header.as_str()).collect();
    if let Some(duplicate) = assertion_slots
        .iter()
        .map(|slot| slot.header.as_str())
        .find(|header| credential_headers.contains(header))
    {
        return Err(format!("callout_credentials: duplicate source header `{duplicate}`").into());
    }
    Ok(())
}

/// Validate a configured source header: parseable, not reserved/framing/hop-by-hop, not
/// routing-prefixed. Mirrors `project_state_owner_headers::parse_projection_header`.
fn parse_credential_source_header(raw: &str) -> Result<HeaderName, FilterError> {
    let name = parse_safe_header(raw)?;
    let lower = name.as_str();
    if RESERVED_SOURCE_PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return Err(format!("callout_credentials: source header `{lower}` uses an internal-trust prefix").into());
    }
    Ok(name)
}

/// Validate an assertion source. Keeping the fixed wire name prevents an
/// assertion slot from becoming an ambient arbitrary-header forwarding path.
fn parse_assertion_source_header(raw: &str) -> Result<HeaderName, FilterError> {
    let name = parse_safe_header(raw)?;
    if name != MCP_AUTHORIZED_HEADER {
        return Err("callout_credentials: assertion source_header must be `x-mcp-authorized`".into());
    }
    Ok(name)
}

/// Parse a non-routing, non-framing request header.
fn parse_safe_header(raw: &str) -> Result<HeaderName, FilterError> {
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

/// Request-scoped maps of per-user callout credentials and authorization assertions.
///
/// Inserted into `RequestExtensions` by the `callout_credentials` filter. Slot ids are safe to
/// log (they come from static config); slot values are secret and never rendered. Keeping the two
/// maps distinct prevents a generic credential consumer from reading an MCP assertion.
#[derive(Default, Clone)]
pub struct CalloutCredentials {
    /// Destination credential values keyed by config-static slot id.
    credential_slots: BTreeMap<String, SecretString>,
    /// MCP authorization assertions keyed by config-static slot id.
    assertion_slots: BTreeMap<String, SecretString>,
}

impl CalloutCredentials {
    /// Create an empty credential set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a secret under `slot`, replacing any prior value for that slot.
    pub fn insert(&mut self, slot: String, value: SecretString) {
        self.credential_slots.insert(slot, value);
    }

    /// Look up the secret staged for `slot`, if any.
    pub fn get(&self, slot: &str) -> Option<&SecretString> {
        self.credential_slots.get(slot)
    }

    /// Store an authorization assertion under `slot`.
    pub(crate) fn insert_assertion(&mut self, slot: String, value: SecretString) {
        self.assertion_slots.insert(slot, value);
    }

    /// Look up the authorization assertion staged for `slot`, if any.
    #[cfg(any(test, feature = "openai-mcp-tools"))]
    pub(crate) fn get_assertion(&self, slot: &str) -> Option<&SecretString> {
        self.assertion_slots.get(slot)
    }

    /// True when no slots are populated.
    pub fn is_empty(&self) -> bool {
        self.credential_slots.is_empty() && self.assertion_slots.is_empty()
    }

    /// Number of populated slots.
    pub fn len(&self) -> usize {
        self.credential_slots.len().saturating_add(self.assertion_slots.len())
    }
}

impl std::fmt::Debug for CalloutCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalloutCredentials")
            .field("credential_slots", &self.credential_slots.keys().collect::<Vec<_>>())
            .field("assertion_slots", &self.assertion_slots.keys().collect::<Vec<_>>())
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
    fn accepts_credential_and_assertion_with_the_same_logical_slot() {
        let f = cfg(
            "credentials:\n  - slot: mcp_gateway\n    source_header: x-user-mcp-key\nassertions:\n  - slot: mcp_gateway\n    source_header: x-mcp-authorized\n",
        );
        assert!(f.is_ok(), "typed slots may share a logical id: {:?}", f.err());
    }

    #[test]
    fn rejects_unknown_field() {
        let f = cfg("credentials:\n  - slot: a\n    source_header: x-user-a\nbogus: 1\n");
        assert!(f.is_err(), "an unknown top-level field must be rejected");
    }

    #[test]
    fn rejects_empty_slots() {
        assert!(
            cfg("credentials: []\nassertions: []\n").is_err(),
            "at least one credential or assertion slot is required"
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
    fn assertion_source_is_fixed_to_mcp_authorized() {
        assert!(
            cfg("assertions:\n  - slot: mcp_gateway\n    source_header: x-mcp-authorized\n").is_ok(),
            "the fixed trusted MCP assertion source must be accepted"
        );
        assert!(
            cfg("assertions:\n  - slot: mcp_gateway\n    source_header: x-user-assertion\n").is_err(),
            "assertions must not become arbitrary forwarded header slots"
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
        creds.insert_assertion("mcp_gateway".to_owned(), SecretString::from("never-print-me"));
        let rendered = format!("{creds:?}");
        assert!(rendered.contains("brave"), "slot name should be visible: {rendered}");
        assert!(
            rendered.contains("mcp_gateway"),
            "assertion slot should be visible: {rendered}"
        );
        assert!(rendered.contains("REDACTED"), "must mark redaction: {rendered}");
        assert!(
            !rendered.contains("super-secret"),
            "secret value must never appear in Debug: {rendered}"
        );
        assert!(!rendered.contains("never-print-me"));
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
            credential_slots: vec![CredentialSlot {
                id: "brave".to_owned(),
                header: HeaderName::from_static("x-user-brave-key"),
            }],
            assertion_slots: Vec::new(),
        }
    }

    /// Make one combined credential + MCP assertion filter.
    fn make_mcp_filter() -> CalloutCredentialsFilter {
        CalloutCredentialsFilter {
            credential_slots: vec![CredentialSlot {
                id: "mcp_gateway".to_owned(),
                header: HeaderName::from_static("x-user-mcp-key"),
            }],
            assertion_slots: vec![CredentialSlot {
                id: "mcp_gateway".to_owned(),
                header: MCP_AUTHORIZED_HEADER,
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
    async fn missing_optional_assertion_is_unpopulated_and_stripped() {
        let filter = make_mcp_filter();
        let request = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(
            ctx.extensions
                .get::<CalloutCredentials>()
                .is_none_or(|secrets| secrets.get_assertion("mcp_gateway").is_none())
        );
        assert!(ctx.request_headers_to_remove.contains(&MCP_AUTHORIZED_HEADER));
    }

    #[tokio::test]
    async fn empty_assertion_is_unpopulated_and_stripped() {
        let filter = make_mcp_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert(MCP_AUTHORIZED_HEADER, HeaderValue::from_static(""));
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(
            ctx.extensions
                .get::<CalloutCredentials>()
                .is_none_or(|secrets| secrets.get_assertion("mcp_gateway").is_none())
        );
        assert!(ctx.request_headers_to_remove.contains(&MCP_AUTHORIZED_HEADER));
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

    #[tokio::test]
    async fn installs_typed_mcp_secrets_and_strips_both_sources() {
        let filter = make_mcp_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert("x-user-mcp-key", HeaderValue::from_static("user-token"));
        request
            .headers
            .insert(MCP_AUTHORIZED_HEADER, HeaderValue::from_static("signed-assertion"));
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let secrets = ctx.extensions.get::<CalloutCredentials>().unwrap();
        assert_eq!(secrets.get("mcp_gateway").unwrap().expose_secret(), "user-token");
        assert_eq!(
            secrets.get_assertion("mcp_gateway").unwrap().expose_secret(),
            "signed-assertion"
        );
        assert!(
            ctx.request_headers_to_remove
                .contains(&HeaderName::from_static("x-user-mcp-key"))
        );
        assert!(ctx.request_headers_to_remove.contains(&MCP_AUTHORIZED_HEADER));
    }

    #[tokio::test]
    async fn duplicate_assertion_is_rejected() {
        let filter = make_mcp_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .append(MCP_AUTHORIZED_HEADER, HeaderValue::from_static("one"));
        request
            .headers
            .append(MCP_AUTHORIZED_HEADER, HeaderValue::from_static("two"));
        let mut ctx = make_filter_context(&request);

        assert!(matches!(
            filter.on_request(&mut ctx).await.unwrap(),
            FilterAction::Reject(_)
        ));
    }

    #[tokio::test]
    async fn oversized_assertion_is_rejected() {
        let filter = make_mcp_filter();
        let mut request = make_request(Method::POST, "/v1/responses");
        request.headers.insert(
            MCP_AUTHORIZED_HEADER,
            HeaderValue::from_bytes(&vec![b'a'; MAX_ASSERTION_BYTES + 1]).unwrap(),
        );
        let mut ctx = make_filter_context(&request);

        assert!(matches!(
            filter.on_request(&mut ctx).await.unwrap(),
            FilterAction::Reject(_)
        ));
    }
}
