// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Credential-safe fingerprints of resolved MCP targets.
//!
//! Resolved tool-map entries carry digests of their effective owner, bearer,
//! and forwarded headers so a later call (an approval resume, or a pooled
//! session reuse) can verify it still targets the same destination with the
//! same identity. Only digests enter the tool map; raw credentials stay in
//! request-scoped secret storage.

use std::{borrow::Cow, collections::HashSet};

use secrecy::{ExposeSecret as _, SecretString};

use crate::{StateOwner, hash::Sha256};

/// Private tool-map field binding approvals to ambient connector identity.
const FORWARDED_HEADERS_FINGERPRINT: &str = "_praxis_forwarded_headers_fingerprint";
/// Private stable-owner binding carried only in the in-memory MCP tool map.
pub(crate) const OWNER_FINGERPRINT: &str = "_praxis_owner_fingerprint";
/// Private effective-bearer binding carried only in the in-memory MCP tool map.
const CREDENTIAL_FINGERPRINT: &str = "_praxis_credential_fingerprint";

/// Compute a stable, credential-safe fingerprint of a resolved MCP target.
///
/// Binds an approval to the concrete destination resolved at approval time: the
/// server URL, request headers, authorization credential, connector id, and a
/// digest of ambient headers forwarded to configured connectors. On resume the
/// proxy recomputes this from the *current* tool-map entry and
/// rejects the call if it differs, so a client cannot keep the approved
/// `(server_label, tool_name)` while redirecting execution elsewhere or swapping
/// credentials.
///
/// A SHA-256 digest — not the raw fields — is emitted so the value is safe to
/// embed in the client-visible, at-rest `mcp_approval_request`: it never
/// discloses the authorization token or trusted identity, and it is deterministic across processes
/// (unlike the std hasher's per-process seed) so the request-time and
/// resume-time computations agree. Fields are length-framed and headers are
/// key-sorted so the digest is independent of JSON key ordering.
///
/// Returns the empty string as a fail-closed sentinel when the headers contain
/// case-insensitive duplicate names: the transport lowercases names and lets the
/// last one in JSON order win, so such a target is ambiguous and must never
/// produce a matchable fingerprint.
pub(crate) fn target_fingerprint(entry: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    for field in [
        "server_url",
        "authorization",
        "connector_id",
        FORWARDED_HEADERS_FINGERPRINT,
        OWNER_FINGERPRINT,
        CREDENTIAL_FINGERPRINT,
    ] {
        hash_segment(&mut hasher, field.as_bytes());
        hash_segment(&mut hasher, &scalar_bytes(entry.get(field)));
    }
    hash_segment(&mut hasher, b"headers");
    if let Some(headers) = entry.get("headers").and_then(serde_json::Value::as_object) {
        // The transport lowercases header names, so two names differing only in
        // case collapse to one and the value actually sent depends on JSON order.
        // A case-sensitive key sort would hash such an ambiguous target to a
        // stable digest that does not reflect what is transmitted, letting a
        // reordered resume reuse the approval with a different value. Fail closed
        // with the empty sentinel instead so it can never match on resume.
        let mut seen = HashSet::with_capacity(headers.len());
        for key in headers.keys() {
            if !seen.insert(key.to_ascii_lowercase()) {
                return String::new();
            }
        }
        let mut keys: Vec<&str> = headers.keys().map(String::as_str).collect();
        keys.sort_unstable();
        for key in keys {
            hash_segment(&mut hasher, key.as_bytes());
            hash_segment(&mut hasher, &scalar_bytes(headers.get(key)));
        }
    }
    hex_digest(hasher.finish())
}

/// Bind a connector entry to its effective request-scoped bearer.
///
/// Only a domain-separated digest enters the in-memory tool map and approval
/// fingerprint. The raw credential remains in request-scoped secret storage.
/// Direct client-selected URLs never retain this private binding.
pub(crate) fn bind_credential_context(entry: &mut serde_json::Value, credential: Option<&SecretString>) {
    let connector = is_connector_tool_entry(entry);
    let Some(object) = entry.as_object_mut() else {
        return;
    };
    if connector {
        if let Some(credential) = credential {
            object.insert(
                CREDENTIAL_FINGERPRINT.to_owned(),
                serde_json::Value::String(credential_fingerprint(credential)),
            );
        } else {
            object.remove(CREDENTIAL_FINGERPRINT);
        }
    } else {
        object.remove(CREDENTIAL_FINGERPRINT);
    }
}

/// Bind a connector entry to the full trusted owner tuple.
///
/// The digest is domain-separated and versioned. Raw identity components never
/// enter approval records or client-visible items.
pub(crate) fn bind_owner_context(entry: &mut serde_json::Value, owner: Option<&StateOwner>) {
    let connector = is_connector_tool_entry(entry);
    let Some(object) = entry.as_object_mut() else {
        return;
    };
    if connector {
        if let Some(owner) = owner {
            object.insert(
                OWNER_FINGERPRINT.to_owned(),
                serde_json::Value::String(owner_fingerprint(owner)),
            );
        } else {
            object.remove(OWNER_FINGERPRINT);
        }
    } else {
        object.remove(OWNER_FINGERPRINT);
    }
}

/// Stable, process-independent digest of `(tenant_id, issuer, subject)`.
pub(crate) fn owner_fingerprint(owner: &StateOwner) -> String {
    let mut hasher = Sha256::new();
    hash_segment(&mut hasher, b"praxis.ai/mcp-owner/v1");
    hash_segment(&mut hasher, owner.tenant_id().as_bytes());
    hash_segment(&mut hasher, owner.issuer().as_bytes());
    hash_segment(&mut hasher, owner.subject().as_bytes());
    hex_digest(hasher.finish())
}

/// Stable digest of the effective connector bearer without retaining it.
fn credential_fingerprint(credential: &SecretString) -> String {
    let mut hasher = Sha256::new();
    hash_segment(&mut hasher, b"praxis.ai/mcp-credential/v1");
    hash_segment(&mut hasher, credential.expose_secret().as_bytes());
    hex_digest(hasher.finish())
}

/// Bind a connector tool-map entry to the exact ambient headers used for calls.
///
/// Only a digest is retained, so trusted identity values do not enter persisted
/// continuation state. Direct client-selected URLs never receive ambient
/// headers and therefore carry no binding.
pub(crate) fn bind_forwarded_header_context(
    entry: &mut serde_json::Value,
    configured_names: &[http::HeaderName],
    headers: &http::HeaderMap,
) {
    let connector = is_connector_tool_entry(entry);
    let Some(object) = entry.as_object_mut() else {
        return;
    };
    if connector {
        object.insert(
            FORWARDED_HEADERS_FINGERPRINT.to_owned(),
            serde_json::Value::String(forwarded_header_fingerprint(configured_names, headers)),
        );
    } else {
        object.remove(FORWARDED_HEADERS_FINGERPRINT);
    }
}

/// Deterministically hash every reserved name and its optional wire values.
fn forwarded_header_fingerprint(configured_names: &[http::HeaderName], headers: &http::HeaderMap) -> String {
    let mut hasher = Sha256::new();
    let mut names: Vec<_> = configured_names.iter().collect();
    names.sort_unstable_by(|left, right| left.as_str().cmp(right.as_str()));
    for name in names {
        hash_segment(&mut hasher, name.as_str().as_bytes());
        for value in headers.get_all(name) {
            hash_segment(&mut hasher, value.as_bytes());
        }
    }
    hex_digest(hasher.finish())
}

/// Encode a SHA-256 digest as lowercase hexadecimal.
fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from_digit(u32::from(*byte >> 4), 16).unwrap_or('0'));
        hex.push(char::from_digit(u32::from(*byte & 0x0F), 16).unwrap_or('0'));
    }
    hex
}

/// Length-framed update so adjacent fields cannot be confused by content that
/// happens to look like a delimiter.
fn hash_segment(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Canonical bytes of an optional JSON scalar for hashing: strings hash as their
/// UTF-8 contents, absent/null as empty, anything else as its compact JSON.
fn scalar_bytes(value: Option<&serde_json::Value>) -> Cow<'_, [u8]> {
    match value {
        None | Some(serde_json::Value::Null) => Cow::Borrowed(&[]),
        Some(serde_json::Value::String(s)) => Cow::Borrowed(s.as_bytes()),
        Some(other) => Cow::Owned(other.to_string().into_bytes()),
    }
}

/// Whether this internally resolved tool entry names a configured connector.
///
/// Optional fields are serialized into the tool map as JSON `null`, so field
/// presence alone is not sufficient to distinguish a request-selected URL.
pub(crate) fn is_connector_tool_entry(entry: &serde_json::Value) -> bool {
    entry
        .get("connector_id")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|connector_id| !connector_id.is_empty())
}
