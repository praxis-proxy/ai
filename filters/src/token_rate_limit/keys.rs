// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Budget-key compilation and per-request resolution (proposal M5 / ai#123).
//!
//! Configured dimensions are compiled once at filter construction, then
//! resolved per request into an opaque backend key. Raw subject IDs,
//! header values, model names, and IP addresses are never stored in
//! Valkey, metrics labels, or filter metadata.

use std::{collections::HashSet, net::IpAddr};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use http::{HeaderMap, header::HeaderName};
use praxis_ai_apis::hash::Sha256;
use praxis_filter::FilterError;

use super::{
    BodyProbe, FALLBACK_KEY, MAX_KEY_LENGTH,
    config::{KeyDimension, KeySpec, MissingKeyPolicy},
};

/// Default header consulted for the `model` dimension when the body has
/// no `model` field. Matches `model_scaled` estimation.
const DEFAULT_MODEL_HEADER: HeaderName = HeaderName::from_static("x-model");

/// Compiled, ready-to-resolve form of [`KeySpec`].
#[derive(Debug, Clone)]
pub(super) struct CompiledKeySpec {
    /// Ordered compiled dimensions.
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
        /// Trusted forwarding header, when configured.
        header: Option<HeaderName>,
    },
    /// Model identity from body then header.
    Model {
        /// Header used when the body has no `model` field.
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

    /// Whether any dimension reads the JSON body (`model`).
    pub(super) fn needs_body(&self) -> bool {
        self.dimensions
            .iter()
            .any(|dim| matches!(dim, CompiledDimension::Model { .. }))
    }

    /// Resolve this spec against one request.
    pub(super) fn resolve(&self, inputs: &KeyInputs<'_>) -> KeyDecision {
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
        KeyDecision::Admit(join_parts(&parts))
    }
}

/// Compile a deserialized [`KeySpec`] into a [`CompiledKeySpec`].
///
/// # Errors
///
/// Empty dimension lists, `global` mixed with other dimensions, duplicate
/// sources, empty/invalid header names.
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

    Ok(CompiledKeySpec {
        dimensions,
        missing: spec.missing,
    })
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
            let key = name.as_str().to_ascii_lowercase();
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
        KeyDimension::Ip { header } => Ok(CompiledDimension::Ip {
            header: optional_header(header, "ip")?,
        }),
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

/// Parse an optional header name, treating blank as unset.
fn optional_header(header: Option<String>, dimension: &str) -> Result<Option<HeaderName>, FilterError> {
    match header {
        Some(name) if name.trim().is_empty() => Ok(None),
        Some(name) => parse_header_name(&name, dimension).map(Some),
        None => Ok(None),
    }
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
        CompiledDimension::AuthenticatedSubject => match nonempty(inputs.subject) {
            Some(subject) => DimensionValue::Part(opaque_part("subject", subject)),
            None => missing_value(spec_missing, 401, "missing_authenticated_subject"),
        },
        CompiledDimension::Ip { header } => match resolve_ip(header.as_ref(), inputs) {
            Some(addr) => DimensionValue::Part(opaque_part("ip", &addr.to_string())),
            None => missing_value(spec_missing, 400, "missing_ip"),
        },
        CompiledDimension::Model { header } => match resolve_model(header, inputs) {
            Some(model) => DimensionValue::Part(opaque_part("model", model)),
            None => missing_value(spec_missing, 400, "missing_model"),
        },
        CompiledDimension::Header { name, missing } => match header_value(inputs.headers, name) {
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

/// Body `model` field, then the configured model header.
fn resolve_model<'a>(header: &HeaderName, inputs: &'a KeyInputs<'a>) -> Option<&'a str> {
    nonempty(inputs.body_probe.and_then(|probe| probe.model.as_deref()))
        .or_else(|| header_value(inputs.headers, header))
}

/// Peer address, or the left-most hop of a configured forwarding header.
fn resolve_ip(header: Option<&HeaderName>, inputs: &KeyInputs<'_>) -> Option<IpAddr> {
    if let Some(name) = header {
        return forwarded_client_ip(header_value(inputs.headers, name));
    }
    inputs.client_addr
}

/// Left-most hop in an `X-Forwarded-For`-style list.
///
/// Accepts a bare IPv4/IPv6 address, or a bracketed IPv6 literal
/// (`[2001:db8::1]`). Invalid tokens fail closed (treated as missing)
/// rather than skipping to a later hop -- the left-most value is the
/// claimed client, and skipping would key on a proxy instead.
fn forwarded_client_ip(raw: Option<&str>) -> Option<IpAddr> {
    let first = raw?.split(',').next()?.trim();
    parse_forwarded_hop(first)
}

/// Parse one XFF hop into an [`IpAddr`].
fn parse_forwarded_hop(token: &str) -> Option<IpAddr> {
    let addr = token
        .strip_prefix('[')
        .and_then(|rest| rest.split(']').next())
        .unwrap_or(token);
    addr.parse().ok()
}

/// Read and trim a header value, treating blank as absent.
fn header_value<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<&'a str> {
    nonempty(headers.get(name).and_then(|value| value.to_str().ok()))
}

/// Trim a string option, dropping empty values.
fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Hash `value` into a fixed-size, non-identifying key part.
pub(super) fn opaque_part(kind: &str, value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    format!("{kind}:v1:{}", URL_SAFE_NO_PAD.encode(digest))
}

/// Hash a header name + value so distinct headers cannot collide.
fn opaque_header_part(header_name: &str, value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(header_name.as_bytes());
    hasher.update(&[0]);
    hasher.update(value.as_bytes());
    format!("hdr:v1:{}", URL_SAFE_NO_PAD.encode(hasher.finish()))
}

/// Join dimension parts, preserving single-part compatibility keys.
fn join_parts(parts: &[String]) -> String {
    if parts.is_empty() {
        return FALLBACK_KEY.to_owned();
    }
    if let [only] = parts {
        return only.clone();
    }
    let joined = parts.join("|");
    if joined.len() <= MAX_KEY_LENGTH {
        joined
    } else {
        opaque_part("comp", &joined)
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
    use super::*;
    use crate::token_rate_limit::config::{KeyDimension, KeySpec, MissingKeyPolicy};

    fn spec(dimensions: Vec<KeyDimension>, missing: MissingKeyPolicy) -> CompiledKeySpec {
        compile_key_spec(KeySpec { dimensions, missing }).unwrap()
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
    fn subject_keys_are_stable_distinct_and_opaque() {
        let compiled = spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        let first = admit(&compiled, &inputs(Some("application-a"), &headers));
        let repeated = admit(&compiled, &inputs(Some("application-a"), &headers));
        let second = admit(&compiled, &inputs(Some("application-b"), &headers));
        assert_eq!(first, repeated);
        assert_ne!(first, second);
        assert!(!first.contains("application-a"));
        assert_eq!(first, opaque_part("subject", "application-a"));
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
        let composite = compiled.resolve(&KeyInputs {
            subject: Some("alice"),
            client_addr: None,
            headers: &headers,
            body_probe: None,
        });
        let subject_only =
            spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject).resolve(&KeyInputs {
                subject: Some("alice"),
                client_addr: None,
                headers: &headers,
                body_probe: None,
            });
        let KeyDecision::Admit(composite) = composite else {
            panic!("expected admit")
        };
        let KeyDecision::Admit(subject_only) = subject_only else {
            panic!("expected admit")
        };
        assert_ne!(composite, subject_only);
        assert!(composite.contains('|'));
    }

    #[test]
    fn body_model_wins_over_header() {
        let compiled = spec(vec![KeyDimension::Model { header: None }], MissingKeyPolicy::Reject);
        let mut headers = HeaderMap::new();
        headers.insert("x-model", "from-header".parse().unwrap());
        let probe = BodyProbe {
            max_tokens: None,
            max_completion_tokens: None,
            model: Some("from-body".into()),
        };
        let KeyDecision::Admit(from_body) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: Some(&probe),
        }) else {
            panic!("expected admit");
        };
        let KeyDecision::Admit(from_header) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: None,
            headers: &headers,
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert_eq!(from_body, opaque_part("model", "from-body"));
        assert_eq!(from_header, opaque_part("model", "from-header"));
        assert_ne!(from_body, from_header);
    }

    #[test]
    fn ip_prefers_parsed_forwarded_header_when_configured() {
        let compiled = spec(
            vec![KeyDimension::Ip {
                header: Some("x-forwarded-for".into()),
            }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.10, 10.0.0.1".parse().unwrap());
        let KeyDecision::Admit(from_header) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert_eq!(from_header, opaque_part("ip", "203.0.113.10"));
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
        assert_eq!(with_model, opaque_part("model", "gpt-4"));

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
    }

    #[test]
    fn invalid_forwarded_header_is_treated_as_missing() {
        let compiled = spec(
            vec![KeyDimension::Ip {
                header: Some("x-forwarded-for".into()),
            }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "not-an-ip, 10.0.0.1".parse().unwrap());
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
        let compiled = spec(
            vec![KeyDimension::Ip {
                header: Some("x-forwarded-for".into()),
            }],
            MissingKeyPolicy::Reject,
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "[2001:db8::10], 10.0.0.1".parse().unwrap());
        let KeyDecision::Admit(key) = compiled.resolve(&KeyInputs {
            subject: None,
            client_addr: Some("127.0.0.1".parse().unwrap()),
            headers: &headers,
            body_probe: None,
        }) else {
            panic!("expected admit");
        };
        assert_eq!(key, opaque_part("ip", "2001:db8::10"));
    }

    #[test]
    fn subject_whitespace_is_trimmed_before_hashing() {
        let compiled = spec(vec![KeyDimension::AuthenticatedSubject], MissingKeyPolicy::Reject);
        let headers = empty_headers();
        let padded = compiled.resolve(&KeyInputs {
            subject: Some("  alice  "),
            client_addr: None,
            headers: &headers,
            body_probe: None,
        });
        let plain = compiled.resolve(&KeyInputs {
            subject: Some("alice"),
            client_addr: None,
            headers: &headers,
            body_probe: None,
        });
        assert_eq!(padded, plain);
        assert_eq!(
            compiled.resolve(&KeyInputs {
                subject: Some("   "),
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
