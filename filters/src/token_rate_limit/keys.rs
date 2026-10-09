// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Budget-key compilation and per-request resolution (proposal M5 / ai#123).
//!
//! Configured dimensions are compiled once at filter construction, then
//! resolved per request into an opaque backend key. Raw subject IDs,
//! header values, model names, and IP addresses are never stored in
//! Valkey, metrics labels, or filter metadata.

use std::{
    collections::HashSet,
    net::{IpAddr, Ipv6Addr, SocketAddr},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use http::{HeaderMap, header::HeaderName};
use praxis_ai_apis::hash::Sha256;
use praxis_filter::FilterError;

use super::{
    BodyProbe, DEFAULT_MODEL_HEADER, FALLBACK_KEY, MAX_KEY_LENGTH,
    config::{KeyDimension, KeySpec, MissingKeyPolicy},
};

/// Compiled, ready-to-resolve form of [`KeySpec`].
#[derive(Debug, Clone)]
pub(super) struct CompiledKeySpec {
    /// Canonical compiled dimensions (duplicates rejected; order is
    /// normalized so YAML reordering does not reset budgets).
    dimensions: Vec<CompiledDimension>,
    /// Default missing-dimension policy.
    missing: MissingKeyPolicy,
}

/// One compiled key dimension.
#[derive(Debug, Clone)]
enum CompiledDimension {
    /// Shared bucket for every matching request.
    Global,
    /// Verified authenticated subject.
    AuthenticatedSubject,
    /// Client IP, optionally taken from a forwarding header.
    Ip {
        /// Forwarding header, when configured.
        header: Option<HeaderName>,
        /// Number of trusted proxy hops to skip from the right of the
        /// forwarding header. `0` selects the right-most hop.
        trusted_hops: u32,
        /// Optional IPv6 prefix length applied before hashing.
        ipv6_prefix: Option<u8>,
    },
    /// Model identity from header, then the JSON body `model` field.
    Model {
        /// Header used when present (default `x-model`).
        header: HeaderName,
    },
    /// Named request header.
    Header {
        /// Header to read.
        name: HeaderName,
        /// Missing-value policy for this header.
        missing: MissingKeyPolicy,
    },
}

/// Outcome of resolving a request onto a budget key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum KeyDecision {
    /// Opaque backend key to reserve against.
    Admit(String),
    /// Reject before provider contact.
    Reject {
        /// HTTP status (401 for missing subject, 400 otherwise).
        status: u16,
        /// Metrics `reason` label; bounded and non-identifying.
        reason: &'static str,
    },
}

/// Request-side inputs the key resolver needs. Kept as a struct so
/// unit tests can drive resolution without constructing a full filter
/// context or a crate-private [`praxis_filter::AuthenticatedIdentity`].
pub(super) struct KeyInputs<'a> {
    /// Verified subject id, when an authentication filter published one.
    pub subject: Option<&'a str>,
    /// Downstream TCP peer, when the protocol layer captured it.
    pub client_addr: Option<IpAddr>,
    /// Request headers.
    pub headers: &'a HeaderMap,
    /// Parsed request-body probe (JSON `model` field), when buffered.
    pub body_probe: Option<&'a BodyProbe>,
}

impl CompiledKeySpec {
    /// A single global bucket -- the historical default.
    #[cfg(test)]
    pub(super) fn global() -> Self {
        Self {
            dimensions: vec![CompiledDimension::Global],
            missing: MissingKeyPolicy::Reject,
        }
    }

    /// Whether any dimension *can* read the JSON body (`model`).
    ///
    /// This does **not** force the filter into `StreamBuffer`. Model
    /// keys prefer the configured header (`x-model` by default) and only
    /// consult the body when the request is already buffered for
    /// estimation.
    #[cfg(test)]
    pub(super) fn reads_model(&self) -> bool {
        self.dimensions
            .iter()
            .any(|dim| matches!(dim, CompiledDimension::Model { .. }))
    }

    /// Resolve this spec against one request.
    pub(super) fn resolve(&self, inputs: &KeyInputs<'_>) -> KeyDecision {
        if self.dimensions.len() == 1 && matches!(self.dimensions.first(), Some(CompiledDimension::Global)) {
            return KeyDecision::Admit(FALLBACK_KEY.to_owned());
        }
        let mut parts = Vec::with_capacity(self.dimensions.len());
        for dimension in &self.dimensions {
            match resolve_dimension(dimension, self.missing, inputs) {
                DimensionValue::Part(part) => parts.push(part),
                DimensionValue::Skip => {},
                DimensionValue::Reject { status, reason } => {
                    return KeyDecision::Reject { status, reason };
                },
            }
        }
        KeyDecision::Admit(join_parts(parts))
    }

    /// Return a stable digest of the compiled key policy.
    ///
    /// This is configuration, not request data: it records the dimensions,
    /// their options, and the missing-value policy so a Valkey writer cannot
    /// reinterpret state created by a differently keyed replica.
    pub(super) fn config_fingerprint(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"praxis:token_rate_limit:key_config");
        digest.update(&[0]);
        digest.update(&[missing_policy_tag(self.missing)]);
        digest.update(&u64::try_from(self.dimensions.len()).unwrap_or(u64::MAX).to_be_bytes());
        for dimension in &self.dimensions {
            digest_dimension(&mut digest, dimension);
        }
        digest.finish().iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// Encode one compiled dimension without including any request-derived value.
fn digest_dimension(digest: &mut Sha256, dimension: &CompiledDimension) {
    match dimension {
        CompiledDimension::Global => digest.update(b"global"),
        CompiledDimension::AuthenticatedSubject => digest.update(b"authenticated_subject"),
        CompiledDimension::Ip {
            header,
            trusted_hops,
            ipv6_prefix,
        } => {
            digest.update(b"ip");
            digest_optional_string(digest, header.as_ref().map(HeaderName::as_str));
            digest.update(&trusted_hops.to_be_bytes());
            match ipv6_prefix {
                Some(prefix) => {
                    digest.update(&[1, *prefix]);
                },
                None => digest.update(&[0]),
            }
        },
        CompiledDimension::Model { header } => {
            digest.update(b"model");
            digest_string(digest, header.as_str());
        },
        CompiledDimension::Header { name, missing } => {
            digest.update(b"header");
            digest_string(digest, name.as_str());
            digest.update(&[missing_policy_tag(*missing)]);
        },
    }
}

/// Length-delimit a string in the key-policy digest.
fn digest_string(digest: &mut Sha256, value: &str) {
    digest.update(&u64::try_from(value.len()).unwrap_or(u64::MAX).to_be_bytes());
    digest.update(value.as_bytes());
}

/// Encode an optional header name without conflating it with an empty value.
fn digest_optional_string(digest: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            digest.update(&[1]);
            digest_string(digest, value);
        },
        None => digest.update(&[0]),
    }
}

/// Compact, stable representation of [`MissingKeyPolicy`].
fn missing_policy_tag(policy: MissingKeyPolicy) -> u8 {
    match policy {
        MissingKeyPolicy::Reject => 0,
        MissingKeyPolicy::Fallback => 1,
    }
}

/// Compile a deserialized [`KeySpec`] into a [`CompiledKeySpec`].
///
/// # Errors
///
/// Empty dimension lists, `global` mixed with other dimensions, duplicate
/// sources, empty/invalid header names, out-of-range `ipv6_prefix`.
pub(super) fn compile_key_spec(spec: KeySpec) -> Result<CompiledKeySpec, FilterError> {
    if spec.dimensions.is_empty() {
        return Err("token_rate_limit: key.dimensions must not be empty".into());
    }

    let has_global = spec.dimensions.iter().any(|dim| matches!(dim, KeyDimension::Global));
    if has_global && spec.dimensions.len() > 1 {
        return Err("token_rate_limit: key dimension 'global' cannot be combined with other dimensions".into());
    }

    let mut seen_named = HashSet::new();
    let mut seen_headers = HashSet::new();
    let mut dimensions = Vec::with_capacity(spec.dimensions.len());
    for dimension in spec.dimensions {
        let compiled = compile_dimension(dimension, spec.missing)?;
        record_dimension(&compiled, &mut seen_named, &mut seen_headers)?;
        dimensions.push(compiled);
    }
    dimensions.sort_by(|left, right| dimension_ord(left).cmp(&dimension_ord(right)));

    Ok(CompiledKeySpec {
        dimensions,
        missing: spec.missing,
    })
}

/// Canonical sort key so composite YAML order cannot silently reset budgets.
fn dimension_ord(dimension: &CompiledDimension) -> (u8, &str) {
    match dimension {
        CompiledDimension::Global => (0, ""),
        CompiledDimension::AuthenticatedSubject => (1, ""),
        CompiledDimension::Ip { .. } => (2, ""),
        CompiledDimension::Model { .. } => (3, ""),
        CompiledDimension::Header { name, .. } => (4, name.as_str()),
    }
}

/// Reject duplicate named sources or duplicate header names.
fn record_dimension(
    compiled: &CompiledDimension,
    seen_named: &mut HashSet<&'static str>,
    seen_headers: &mut HashSet<String>,
) -> Result<(), FilterError> {
    match compiled {
        CompiledDimension::Global => insert_unique(seen_named, "global"),
        CompiledDimension::AuthenticatedSubject => insert_unique(seen_named, "authenticated_subject"),
        CompiledDimension::Ip { .. } => insert_unique(seen_named, "ip"),
        CompiledDimension::Model { .. } => insert_unique(seen_named, "model"),
        CompiledDimension::Header { name, .. } => {
            // `HeaderName` is already lowercase.
            let key = name.as_str().to_owned();
            if seen_headers.insert(key) {
                Ok(())
            } else {
                Err(format!("token_rate_limit: duplicate key header dimension '{}'", name.as_str()).into())
            }
        },
    }
}

/// Compile one deserialized dimension, validating header names.
fn compile_dimension(
    dimension: KeyDimension,
    spec_missing: MissingKeyPolicy,
) -> Result<CompiledDimension, FilterError> {
    match dimension {
        KeyDimension::Global => Ok(CompiledDimension::Global),
        KeyDimension::AuthenticatedSubject => Ok(CompiledDimension::AuthenticatedSubject),
        KeyDimension::Ip {
            header,
            trusted_hops,
            ipv6_prefix,
        } => compile_ip_dimension(header, trusted_hops, ipv6_prefix),
        KeyDimension::Model { header } => Ok(CompiledDimension::Model {
            header: match header {
                Some(name) => parse_header_name(&name, "model")?,
                None => DEFAULT_MODEL_HEADER,
            },
        }),
        KeyDimension::Header { name, missing } => {
            if name.trim().is_empty() {
                return Err("token_rate_limit: key header name must not be empty".into());
            }
            Ok(CompiledDimension::Header {
                name: parse_header_name(&name, "header")?,
                missing: missing.unwrap_or(spec_missing),
            })
        },
    }
}

/// Compile an IP dimension, validating the header name and prefix length.
fn compile_ip_dimension(
    header: Option<String>,
    trusted_hops: u32,
    ipv6_prefix: Option<u8>,
) -> Result<CompiledDimension, FilterError> {
    if ipv6_prefix.is_some_and(|prefix| !(1..=128).contains(&prefix)) {
        return Err("token_rate_limit: ip.ipv6_prefix must be between 1 and 128".into());
    }
    Ok(CompiledDimension::Ip {
        header: match header {
            Some(name) => Some(parse_header_name(&name, "ip")?),
            None => None,
        },
        trusted_hops,
        ipv6_prefix,
    })
}

/// Parse a header name used as a key dimension.
fn parse_header_name(name: &str, dimension: &str) -> Result<HeaderName, FilterError> {
    HeaderName::try_from(name.trim()).map_err(|error| {
        FilterError::from(format!(
            "token_rate_limit: invalid {dimension} header name '{name}': {error}"
        ))
    })
}

/// Insert `name` into `seen`, erroring on duplicates.
fn insert_unique(seen: &mut HashSet<&'static str>, name: &'static str) -> Result<(), FilterError> {
    if seen.insert(name) {
        Ok(())
    } else {
        Err(format!("token_rate_limit: duplicate key dimension '{name}'").into())
    }
}

/// Per-dimension resolution outcome.
enum DimensionValue {
    /// Include this opaque part in the composed key.
    Part(String),
    /// Drop this dimension (`missing: fallback`).
    Skip,
    /// Fail closed.
    Reject {
        /// HTTP status to return.
        status: u16,
        /// Metrics reason label.
        reason: &'static str,
    },
}

/// Resolve one compiled dimension against the request.
fn resolve_dimension(
    dimension: &CompiledDimension,
    spec_missing: MissingKeyPolicy,
    inputs: &KeyInputs<'_>,
) -> DimensionValue {
    match dimension {
        CompiledDimension::Global => DimensionValue::Part(FALLBACK_KEY.to_owned()),
        CompiledDimension::AuthenticatedSubject => match inputs.subject.filter(|subject| !subject.is_empty()) {
            Some(subject) => DimensionValue::Part(opaque_part("subject", subject.as_bytes())),
            None => missing_value(spec_missing, 401, "missing_authenticated_subject"),
        },
        CompiledDimension::Ip {
            header,
            trusted_hops,
            ipv6_prefix,
        } => match resolve_ip(header.as_ref(), *trusted_hops, *ipv6_prefix, inputs) {
            Some(canonical) => DimensionValue::Part(opaque_part("ip", canonical.as_bytes())),
            None => missing_value(spec_missing, 400, "missing_ip"),
        },
        CompiledDimension::Model { header } => match resolve_model(header, inputs) {
            Some(model) => DimensionValue::Part(opaque_part("model", model.as_bytes())),
            None => missing_value(spec_missing, 400, "missing_model"),
        },
        CompiledDimension::Header { name, missing } => match header_bytes(inputs.headers, name) {
            Some(value) => DimensionValue::Part(opaque_header_part(name.as_str(), value)),
            None => missing_value(*missing, 400, "missing_header"),
        },
    }
}

/// Apply the missing-dimension policy.
fn missing_value(policy: MissingKeyPolicy, status: u16, reason: &'static str) -> DimensionValue {
    match policy {
        MissingKeyPolicy::Reject => DimensionValue::Reject { status, reason },
        MissingKeyPolicy::Fallback => DimensionValue::Skip,
    }
}

/// Model header first, then the JSON body `model` field when already buffered.
fn resolve_model<'a>(header: &HeaderName, inputs: &'a KeyInputs<'a>) -> Option<&'a str> {
    header_value(inputs.headers, header)
        .or_else(|| nonempty(inputs.body_probe.and_then(|probe| probe.model.as_deref())))
}

/// Peer address, or a configured forwarding header when that header is present.
///
/// When a forwarding header is configured, a missing header falls back to
/// the TCP peer (health checks, in-cluster clients). A present but
/// unusable header (too few hops, unparseable selected hop) is treated
/// as missing rather than skipping to another hop.
fn resolve_ip(
    header: Option<&HeaderName>,
    trusted_hops: u32,
    ipv6_prefix: Option<u8>,
    inputs: &KeyInputs<'_>,
) -> Option<String> {
    let addr = if let Some(name) = header {
        match forwarded_header_raw(inputs.headers, name) {
            Some(raw) => forwarded_client_ip(&raw, trusted_hops)?,
            None => inputs.client_addr?,
        }
    } else {
        inputs.client_addr?
    };
    Some(canonicalize_ip(addr, ipv6_prefix))
}

/// Join every `HeaderMap` field line for `name` (proxies often split XFF).
fn forwarded_header_raw(headers: &HeaderMap, name: &HeaderName) -> Option<String> {
    let mut joined = String::new();
    for value in headers.get_all(name) {
        let Ok(text) = value.to_str() else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if !joined.is_empty() {
            joined.push(',');
        }
        joined.push_str(text);
    }
    (!joined.is_empty()).then_some(joined)
}

/// Right-most hop after `trusted_hops` trusted proxies.
///
/// Appending proxies (ALB, nginx `$proxy_add_x_forwarded_for`, Envoy)
/// put the client-controlled values on the left. `trusted_hops = 0`
/// therefore selects the right-most hop -- the address the last proxy
/// appended. `trusted_hops = 1` skips that last hop (the immediate
/// proxy) and takes the next address to its left.
fn forwarded_client_ip(raw: &str, trusted_hops: u32) -> Option<IpAddr> {
    let hops: Vec<&str> = raw.split(',').map(str::trim).filter(|hop| !hop.is_empty()).collect();
    let skip = usize::try_from(trusted_hops).ok()?;
    let index = hops.len().checked_sub(skip.saturating_add(1))?;
    parse_forwarded_hop(hops.get(index)?)
}

/// Parse one XFF hop into an [`IpAddr`].
///
/// Accepts a bare IPv4/IPv6 address, `[v6]`, `[v6]:port`, or `v4:port`.
fn parse_forwarded_hop(token: &str) -> Option<IpAddr> {
    if let Ok(socket) = token.parse::<SocketAddr>() {
        return Some(normalize_mapped_ipv4(socket.ip()));
    }
    let addr = token
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(token);
    addr.parse().ok().map(normalize_mapped_ipv4)
}

/// Collapse IPv4-mapped IPv6 addresses so dual-stack views of the same
/// client share a bucket.
fn normalize_mapped_ipv4(addr: IpAddr) -> IpAddr {
    match addr {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(addr, IpAddr::V4),
        IpAddr::V4(_) => addr,
    }
}

/// Apply optional IPv6 prefix masking, then render the hash input.
fn canonicalize_ip(addr: IpAddr, ipv6_prefix: Option<u8>) -> String {
    let addr = normalize_mapped_ipv4(addr);
    match (addr, ipv6_prefix) {
        (IpAddr::V6(v6), Some(prefix)) => format!("{}/{}", mask_ipv6(v6, prefix), prefix),
        (addr, _) => addr.to_string(),
    }
}

/// Zero host bits past `prefix` on an IPv6 address.
fn mask_ipv6(addr: Ipv6Addr, prefix: u8) -> Ipv6Addr {
    let prefix = prefix.min(128);
    if prefix == 128 {
        return addr;
    }
    let mut segments = addr.segments();
    let mut remaining = prefix;
    for segment in &mut segments {
        if remaining >= 16 {
            remaining -= 16;
            continue;
        }
        if remaining == 0 {
            *segment = 0;
            continue;
        }
        *segment &= !0_u16 << (16 - remaining);
        remaining = 0;
    }
    Ipv6Addr::from(segments)
}

/// Read and trim a header value as UTF-8, treating blank as absent.
fn header_value<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<&'a str> {
    nonempty(headers.get(name).and_then(|value| value.to_str().ok()))
}

/// Read a header as raw bytes so non-ASCII values still key a bucket.
fn header_bytes<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<&'a [u8]> {
    let value = headers.get(name)?;
    let trimmed = value.as_bytes().trim_ascii();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Trim a string option, dropping empty values.
fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Hash `value` into a fixed-size, non-identifying key part.
pub(super) fn opaque_part(kind: &str, value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    format!("{kind}:v1:{}", URL_SAFE_NO_PAD.encode(digest))
}

/// Hash a header name + value so distinct headers cannot collide.
fn opaque_header_part(header_name: &str, value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(header_name.as_bytes());
    hasher.update(&[0]);
    hasher.update(value);
    format!("hdr:v1:{}", URL_SAFE_NO_PAD.encode(hasher.finish()))
}

/// Join dimension parts, preserving single-part compatibility keys.
fn join_parts(mut parts: Vec<String>) -> String {
    match parts.len() {
        0 => FALLBACK_KEY.to_owned(),
        1 => parts.remove(0),
        _ => {
            let joined = parts.join("|");
            if joined.len() <= MAX_KEY_LENGTH {
                joined
            } else {
                opaque_part("comp", joined.as_bytes())
            }
        },
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use http::HeaderValue;

    use super::*;
    use crate::token_rate_limit::config::{KeyDimension, KeySpec, MissingKeyPolicy};

    /// Known-answer SHA-256 / URL-safe base64 of `application-a`.
    const SUBJECT_APPLICATION_A: &str = "subject:v1:0wfYjuGrt6TOMloboGk1H_QGJBDgt-0kQ-OyQ3qjy34";
    /// Known-answer for IPv4 `203.0.113.10`.
    const IP_203_0_113_10: &str = "ip:v1:Yx8IFAsktydNEt88N6GoDOWHba_XAH13LgEU_d-ItoI";
    /// Known-answer for model `gpt-4`.
    const MODEL_GPT_4: &str = "model:v1:GxrnBpcWMq2l9ts2VO2kUUjyT6wPerxVwZVlSUpTYdo";
    /// Known-answer for header `x-tenant-id` / `acme`.
    const HDR_TENANT_ACME: &str = "hdr:v1:TzI_tLV4WcGycPF8tnAf4jkE8iUTijVowQfcgPx3vss";
    /// Known-answer composite of subject `alice` and model `gpt-4`.
    const COMPOSITE_ALICE_GPT4: &str =
        "subject:v1:K9gGyX8OAK8aH8Myj6djqSaXI8jbj6xPk69x2xhtbpA|model:v1:GxrnBpcWMq2l9ts2VO2kUUjyT6wPerxVwZVlSUpTYdo";

    fn spec(dimensions: Vec<KeyDimension>, missing: MissingKeyPolicy) -> CompiledKeySpec {
        compile_key_spec(KeySpec { dimensions, missing }).unwrap()
    }

    fn ip_dim(header: Option<&str>) -> KeyDimension {
        KeyDimension::Ip {
            header: header.map(str::to_owned),
            trusted_hops: 0,
            ipv6_prefix: None,
        }
    }

    fn empty_headers() -> HeaderMap {
        HeaderMap::new()
    }

    fn inputs<'a>(subject: Option<&'a str>, headers: &'a HeaderMap) -> KeyInputs<'a> {
        KeyInputs {
            subject,
            client_addr: None,
            headers,
            body_probe: None,
        }
    }

    fn admit(compiled: &CompiledKeySpec, inputs: &KeyInputs<'_>) -> String {
        match compiled.resolve(inputs) {
            KeyDecision::Admit(key) => key,
            KeyDecision::Reject { status, reason } => {
                panic!("expected admit, got reject status={status} reason={reason}")
            },
        }
    }

    #[test]
    fn global_is_the_legacy_fallback_key() {
        let compiled = CompiledKeySpec::global();
        let headers = empty_headers();
        let decision = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: None,
        });
        assert_eq!(decision, KeyDecision::Admit(FALLBACK_KEY.to_owned()));
    }

    #[test]
    fn config_fingerprint_captures_key_policy_and_canonicalizes_order() {
        let global = CompiledKeySpec::global().config_fingerprint();
        let subject = spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject);
        let subject_reversed = spec(
            vec![KeyDimension::Model { header: None }, KeyDimension::AuthenticatedSubject],
            MissingKeyPolicy::Reject,
        );
        let subject_forward = spec(
            vec![KeyDimension::AuthenticatedSubject, KeyDimension::Model { header: None }],
            MissingKeyPolicy::Reject,
        );

        assert_ne!(
            global,
            subject.config_fingerprint(),
            "global and subject keying must not share a marker"
        );
        assert_eq!(
            subject_reversed.config_fingerprint(),
            subject_forward.config_fingerprint(),
            "composite declaration order is not semantic"
        );
        assert_ne!(
            subject.config_fingerprint(),
            spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Fallback).config_fingerprint(),
            "missing-dimension policy is part of the shared accounting contract"
        );
    }

    #[test]
    fn subject_keys_are_stable_distinct_and_opaque() {
        let compiled = spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        let first = admit(&compiled, &inputs(Some("application-a"), &headers));
        let repeated = admit(&compiled, &inputs(Some("application-a"), &headers));
        let second = admit(&compiled, &inputs(Some("application-b"), &headers));
        assert_eq!(first, repeated);
        assert_ne!(first, second);
        assert!(!first.contains("application-a"));
        assert_eq!(first, SUBJECT_APPLICATION_A);
        assert_eq!(first, opaque_part("subject", b"application-a"));
    }

    #[test]
    fn known_answer_hashes_pin_persisted_key_format() {
        assert_eq!(opaque_part("ip", b"203.0.113.10"), IP_203_0_113_10);
        assert_eq!(opaque_part("model", b"gpt-4"), MODEL_GPT_4);
        assert_eq!(opaque_header_part("x-tenant-id", b"acme"), HDR_TENANT_ACME);
        let compiled = spec(
            vec![KeyDimension::AuthenticatedSubject, KeyDimension::Model { header: None }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "gpt-4".parse().unwrap());
        assert_eq!(admit(&compiled, &inputs(Some("alice"), &headers)), COMPOSITE_ALICE_GPT4);
    }

    #[test]
    fn missing_subject_rejects_with_401() {
        let compiled = spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        assert_eq!(
            compiled.resolve(&KeyInputs {
                subject: None,
                client_addr: None,
                headers: &headers,
                body_probe: None,
            }),
            KeyDecision::Reject {
                status: 401,
                reason: "missing_authenticated_subject",
            }
        );
    }

    #[test]
    fn missing_header_can_fall_back_to_the_global_bucket() {
        let compiled = spec(
            vec![KeyDimension::Header {
                name: "x-tenant-id".into(),
                missing: Some(MissingKeyPolicy::Fallback),
            }],
            MissingKeyPolicy::Reject,
        );
        let headers = empty_headers();
        assert_eq!(
            compiled.resolve(&KeyInputs {
                subject: None,
                client_addr: None,
                headers: &headers,
                body_probe: None,
            }),
            KeyDecision::Admit(FALLBACK_KEY.to_owned())
        );
    }

    #[test]
    fn composite_subject_and_model_are_independent_of_either_alone() {
        let compiled = spec(
            vec![KeyDimension::AuthenticatedSubject, KeyDimension::Model { header: None }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "gpt-4".parse().unwrap());
        let composite = admit(&compiled, &inputs(Some("alice"), &headers));
        let subject_only = admit(
            &spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject),
            &inputs(Some("alice"), &headers),
        );
        assert_ne!(composite, subject_only);
        assert!(composite.contains('|'));
        assert_eq!(composite, COMPOSITE_ALICE_GPT4);
    }

    #[test]
    fn composite_yaml_order_does_not_change_the_resolved_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "gpt-4".parse().unwrap());
        let forward = spec(
            vec![KeyDimension::AuthenticatedSubject, KeyDimension::Model { header: None }],
            MissingKeyPolicy::Reject,
        );
        let reverse = spec(
            vec![KeyDimension::Model { header: None }, KeyDimension::AuthenticatedSubject],
            MissingKeyPolicy::Reject,
        );
        assert_eq!(
            admit(&forward, &inputs(Some("alice"), &headers)),
            admit(&reverse, &inputs(Some("alice"), &headers))
        );
    }

    #[test]
    fn model_header_wins_over_body() {
        let compiled = spec(vec![KeyDimension::Model { header: None }], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "from-header".parse().unwrap());
        let probe = BodyProbe {
            max_tokens: None,
            max_completion_tokens: None,
            model: Some("from-body".into()),
        };
        let from_both = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: Some(&probe),
        });
        let from_header = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(from_both) = from_both else {
            panic!("expected admit");
        };
        let KeyDecision::Admit(from_header) = from_header else {
            panic!("expected admit");
        };
        assert_eq!(from_both, opaque_part("model", b"from-header"));
        assert_eq!(from_header, opaque_part("model", b"from-header"));
        assert!(compiled.reads_model());
        let empty = empty_headers();
        let from_body = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &empty,
            body_probe: Some(&probe),
        });
        let KeyDecision::Admit(from_body) = from_body else {
            panic!("expected admit");
        };
        assert_eq!(from_body, opaque_part("model", b"from-body"));
        assert_ne!(from_both, from_body);
    }

    #[test]
    fn ip_uses_rightmost_forwarded_hop_by_default() {
        let compiled = spec(vec![ip_dim(Some("x-forwarded-for"))], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.10, 10.0.0.1".parse().unwrap());
        let from_header = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(from_header) = from_header else {
            panic!("expected admit");
        };
        assert_eq!(from_header, opaque_part("ip", b"10.0.0.1"));
    }

    #[test]
    fn ip_trusted_hops_skips_from_the_right() {
        let compiled = spec(
            vec![KeyDimension::Ip {
                header: Some("x-forwarded-for".into()),
                trusted_hops: 1,
                ipv6_prefix: None,
            }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.10, 10.0.0.1".parse().unwrap());
        let key = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(key) = key else {
            panic!("expected admit");
        };
        assert_eq!(key, IP_203_0_113_10);
    }

    #[test]
    fn ip_joins_all_forwarded_header_lines() {
        let compiled = spec(vec![ip_dim(Some("x-forwarded-for"))], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.append("x-forwarded-for", "203.0.113.10".parse().unwrap());
        headers.append("x-forwarded-for", "10.0.0.1".parse().unwrap());
        let key = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(key) = key else {
            panic!("expected admit");
        };
        assert_eq!(key, opaque_part("ip", b"10.0.0.1"));
    }

    #[test]
    fn ip_parses_v4_with_port() {
        let compiled = spec(vec![ip_dim(Some("x-forwarded-for"))], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.10:54321".parse().unwrap());
        let key = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(key) = key else {
            panic!("expected admit");
        };
        assert_eq!(key, IP_203_0_113_10);
    }

    #[test]
    fn ip_falls_back_to_peer_when_forwarding_header_is_absent() {
        let compiled = spec(vec![ip_dim(Some("x-forwarded-for"))], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        let key = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("203.0.113.10".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(key) = key else {
            panic!("expected admit");
        };
        assert_eq!(key, IP_203_0_113_10);
    }

    #[test]
    fn ip_normalizes_ipv4_mapped_addresses() {
        let compiled = spec(vec![ip_dim(None)], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        let v4 = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("203.0.113.7".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let mapped = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("::ffff:203.0.113.7".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(v4) = v4 else {
            panic!("expected admit");
        };
        let KeyDecision::Admit(mapped) = mapped else {
            panic!("expected admit");
        };
        assert_eq!(v4, mapped);
        assert_eq!(v4, opaque_part("ip", b"203.0.113.7"));
    }

    #[test]
    fn ip_ipv6_prefix_masks_host_bits() {
        let compiled = spec(
            vec![KeyDimension::Ip {
                header: None,
                trusted_hops: 0,
                ipv6_prefix: Some(64),
            }],
            MissingKeyPolicy::Reject,
        );
        let headers = empty_headers();
        let first = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("2001:db8:1:2:aaaa::1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let second = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("2001:db8:1:2:bbbb::2".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        });
        let KeyDecision::Admit(first) = first else {
            panic!("expected admit");
        };
        let KeyDecision::Admit(second) = second else {
            panic!("expected admit");
        };
        assert_eq!(first, second);
        assert_eq!(first, opaque_part("ip", b"2001:db8:1:2::/64"));
    }

    #[test]
    fn rejects_global_combined_with_other_dimensions() {
        let err = compile_key_spec(KeySpec {
            dimensions: vec![KeyDimension::Global, KeyDimension::AuthenticatedSubject],
            missing: MissingKeyPolicy::Reject,
        })
        .unwrap_err();
        assert!(err.to_string().contains("cannot be combined"), "got: {err}");
    }

    #[test]
    fn rejects_duplicate_header_dimensions() {
        let err = compile_key_spec(KeySpec {
            dimensions: vec![
                KeyDimension::Header {
                    name: "X-Tenant-Id".into(),
                    missing: None,
                },
                KeyDimension::Header {
                    name: "x-tenant-id".into(),
                    missing: None,
                },
            ],
            missing: MissingKeyPolicy::Reject,
        })
        .unwrap_err();
        assert!(err.to_string().contains("duplicate key header"), "got: {err}");
    }

    #[test]
    fn rejects_blank_ip_forwarding_header() {
        let err = compile_key_spec(KeySpec {
            dimensions: vec![ip_dim(Some("   "))],
            missing: MissingKeyPolicy::Reject,
        })
        .unwrap_err();
        assert!(err.to_string().contains("invalid ip header name"), "got: {err}");
    }

    #[test]
    fn composite_fallback_drops_missing_dimensions_and_keeps_the_rest() {
        let compiled = spec(
            vec![KeyDimension::AuthenticatedSubject, KeyDimension::Model { header: None }],
            MissingKeyPolicy::Fallback,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "gpt-4".parse().unwrap());
        let KeyDecision::Admit(with_model) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert_eq!(with_model, MODEL_GPT_4);

        let KeyDecision::Admit(neither) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &empty_headers(),
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert_eq!(neither, FALLBACK_KEY);
    }

    #[test]
    fn per_header_missing_override_rejects_even_when_spec_falls_back() {
        let compiled = spec(
            vec![
                KeyDimension::Header {
                    name: "x-api-key".into(),
                    missing: Some(MissingKeyPolicy::Reject),
                },
                KeyDimension::Model { header: None },
            ],
            MissingKeyPolicy::Fallback,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "gpt-4".parse().unwrap());
        assert_eq!(
            compiled.resolve(&KeyInputs {
                subject: None,
                client_addr: None,
                headers: &headers,
                body_probe: None,
            }),
            KeyDecision::Reject {
                status: 400,
                reason: "missing_header",
            }
        );
    }

    #[test]
    fn distinct_headers_with_the_same_value_do_not_collide() {
        let header_key = |name: &str| {
            spec(
                vec![KeyDimension::Header {
                    name: name.into(),
                    missing: None,
                }],
                MissingKeyPolicy::Reject,
            )
        };
        let mut tenant_headers = HeaderMap::new();
        tenant_headers.insert("x-tenant-id", "acme".parse().unwrap());
        let mut team_headers = HeaderMap::new();
        team_headers.insert("x-team-id", "acme".parse().unwrap());
        let tenant = admit(&header_key("x-tenant-id"), &inputs(None, &tenant_headers));
        let team = admit(&header_key("x-team-id"), &inputs(None, &team_headers));
        assert_ne!(tenant, team);
        assert_eq!(tenant, HDR_TENANT_ACME);
    }

    #[test]
    fn header_keys_hash_non_ascii_values() {
        let compiled = spec(
            vec![KeyDimension::Header {
                name: "x-tenant-id".into(),
                missing: None,
            }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-tenant-id",
            HeaderValue::from_bytes("Ünïcode-Tenant".as_bytes()).unwrap(),
        );
        let key = admit(&compiled, &inputs(None, &headers));
        assert!(key.starts_with("hdr:v1:"));
        assert_eq!(key, opaque_header_part("x-tenant-id", "Ünïcode-Tenant".as_bytes()));
    }

    #[test]
    fn invalid_forwarded_header_is_treated_as_missing() {
        let compiled = spec(vec![ip_dim(Some("x-forwarded-for"))], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "10.0.0.1, not-an-ip".parse().unwrap());
        assert_eq!(
            compiled.resolve(&KeyInputs {
                subject: None,
                client_addr: Some("127.0.0.1".parse().unwrap()),
                headers: &headers,
                body_probe: None,
            }),
            KeyDecision::Reject {
                status: 400,
                reason: "missing_ip",
            }
        );
    }

    #[test]
    fn bracketed_ipv6_forwarded_header_is_parsed() {
        let compiled = spec(vec![ip_dim(Some("x-forwarded-for"))], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "[2001:db8::10]".parse().unwrap());
        let KeyDecision::Admit(key) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert_eq!(key, opaque_part("ip", b"2001:db8::10"));
    }

    #[test]
    fn authenticated_subject_is_hashed_without_trimming() {
        let compiled = spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        let padded = admit(&compiled, &inputs(Some("  alice  "), &headers));
        let plain = admit(&compiled, &inputs(Some("alice"), &headers));
        assert_ne!(padded, plain);
        assert_eq!(plain, opaque_part("subject", b"alice"));
        assert_eq!(padded, opaque_part("subject", b"  alice  "));
        let whitespace_only = admit(&compiled, &inputs(Some("   "), &headers));
        assert_eq!(whitespace_only, opaque_part("subject", b"   "));
    }

    #[test]
    fn rejects_duplicate_named_dimensions() {
        let err = compile_key_spec(KeySpec {
            dimensions: vec![
                KeyDimension::Model { header: None },
                KeyDimension::Model {
                    header: Some("x-model".into()),
                },
            ],
            missing: MissingKeyPolicy::Reject,
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("duplicate key dimension 'model'"),
            "got: {err}"
        );
    }

    #[test]
    fn overlong_composites_fold_to_a_bounded_digest() {
        let many_headers = (0..12)
            .map(|i| KeyDimension::Header {
                name: format!("x-custom-dimension-{i}"),
                missing: None,
            })
            .collect();
        let compiled = spec(many_headers, MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        for i in 0..12 {
            headers.insert(
                HeaderName::try_from(format!("x-custom-dimension-{i}")).unwrap(),
                "value".parse().unwrap(),
            );
        }
        let KeyDecision::Admit(key) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert!(key.starts_with("comp:v1:"));
        assert!(key.len() <= MAX_KEY_LENGTH);
    }
}
