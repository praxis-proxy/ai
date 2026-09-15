// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! A shared helper for body-derived data promotion.

use praxis_filter::{
    FilterError,
    builtins::http::{
        payload_processing::config_validation::validate_header_name, value_safety::is_safe_promoted_value,
    },
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Longest value that may be promoted to a header, metadata key, or filter result.
pub const MAX_PROMOTED_VALUE_LEN: usize = 256;

// -----------------------------------------------------------------------------
// is_promotable_value
// -----------------------------------------------------------------------------

/// Returns `true` iff `val` is within the length limit and safe for HTTP header use.
pub fn is_promotable_value(val: &str) -> bool {
    val.len() <= MAX_PROMOTED_VALUE_LEN && is_safe_promoted_value(val)
}

/// Namespaces that classification filters may overwrite when the call
/// site does not pin a single dedicated header.
const AI_FACT_PREFIXES: &[&str] = &["x-praxis-ai-", "x-praxis-responses-"];

/// Dedicated model-identity headers that model rewrite may overwrite.
const MODEL_IDENTITY_HEADERS: &[&str] = &["x-praxis-ai-effective-model", "x-praxis-ai-original-model"];

/// Hop-by-hop, framing, Host, and proxy-auth names that must not be
/// used as promotion-header targets.
///
/// Composes [`crate::http_hop::is_hop_by_hop`] with `Host` and
/// `Content-Length`, which are transport-controlled but not hop-by-hop.
pub fn is_transport_controlled_header(name: &str) -> bool {
    is_transport_controlled_header_lowercase(&name.to_ascii_lowercase())
}

/// Like [`is_transport_controlled_header`], but `name` must already be a
/// lowercase HTTP field name, as produced by
/// [`http::HeaderName::as_str`].
#[must_use]
pub fn is_transport_controlled_header_lowercase(name: &str) -> bool {
    name == "content-length" || name == "host" || crate::http_hop::is_hop_by_hop(name)
}

/// Whether `name` is known to carry credentials or provider API keys.
///
/// Shared with inference-fixture sanitization so promotion targets and
/// recorded-header stripping cannot drift.
pub fn is_credential_header(name: &str) -> bool {
    is_credential_header_lowercase(&name.to_ascii_lowercase())
}

/// Like [`is_credential_header`], but `name` must already be lowercase.
fn is_credential_header_lowercase(name: &str) -> bool {
    matches!(
        name,
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "set-cookie"
            | "www-authenticate"
            | "x-api-key"
            | "api-key"
            | "x-goog-api-key"
    )
}

/// Whether `name` is unsafe as a body-derived promotion-header target.
///
/// Blocks transport-controlled names, request credentials (including
/// provider API-key headers), and internal `x-praxis-*` headers outside
/// the AI fact namespaces (`x-praxis-ai-*`, `x-praxis-responses-*`).
pub fn is_unsafe_promotion_header(name: &str) -> bool {
    is_unsafe_promotion_target(name, &[], AI_FACT_PREFIXES)
}

/// Whether `name` is unsafe as a client-derived model-identity promotion target.
///
/// In addition to transport and credential names, this rejects other
/// Praxis classification and routing facts such as `x-praxis-ai-format`.
pub fn is_unsafe_model_identity_promotion_header(name: &str) -> bool {
    is_unsafe_promotion_target(name, MODEL_IDENTITY_HEADERS, &[])
}

/// Validate syntax and reject unsafe promotion-header targets.
///
/// Allows dedicated AI fact namespaces (`x-praxis-ai-*`,
/// `x-praxis-responses-*`) and custom non-`x-praxis-*` headers.
///
/// # Errors
///
/// Returns [`FilterError`] when the name is empty, not a valid HTTP
/// header name, or an unsafe promotion target.
pub fn validate_promotion_header(filter: &str, field: &str, name: Option<&str>) -> Result<(), FilterError> {
    validate_header_name(filter, field, name)?;
    reject_unsafe_promotion_target(filter, field, name, &[], AI_FACT_PREFIXES)
}

/// Validate a header that will receive a client-derived model string.
///
/// # Errors
///
/// Returns [`FilterError`] when the name is empty, not a valid HTTP
/// header name, or would overwrite transport, credential, or other
/// Praxis routing state.
pub fn validate_model_identity_promotion_header(
    filter: &str,
    field: &str,
    name: Option<&str>,
) -> Result<(), FilterError> {
    validate_dedicated_promotion_header(filter, field, name, MODEL_IDENTITY_HEADERS)
}

/// Validate a header that may overwrite only `dedicated` internal names.
///
/// Custom non-`x-praxis-*` names remain allowed. Any other `x-praxis-*`
/// name, including other AI classification facts, is rejected.
///
/// Pass an empty `dedicated` slice when the filter has no reserved
/// internal header (for example `model_to_header`'s default `X-Model`).
///
/// # Errors
///
/// Returns [`FilterError`] when the name is empty, not a valid HTTP
/// header name, or an unsafe promotion target.
pub fn validate_dedicated_promotion_header(
    filter: &str,
    field: &str,
    name: Option<&str>,
    dedicated: &[&str],
) -> Result<(), FilterError> {
    validate_header_name(filter, field, name)?;
    reject_unsafe_promotion_target(filter, field, name, dedicated, &[])
}

/// Reject two configured promotion fields that resolve to the same header.
///
/// `None` entries are skipped. Comparison is ASCII case-insensitive.
///
/// # Errors
///
/// Returns [`FilterError`] when two named fields share a header.
pub fn reject_duplicate_promotion_fields(filter: &str, fields: &[(&str, Option<&str>)]) -> Result<(), FilterError> {
    let mut seen: Vec<(&str, String)> = Vec::new();
    for &(field, name) in fields {
        let Some(raw) = name else {
            continue;
        };
        let normalized = raw.to_ascii_lowercase();
        if let Some(&(other, _)) = seen.iter().find(|(_, existing)| existing == &normalized) {
            return Err(format!("{filter}: '{field}' and '{other}' must not use the same header name").into());
        }
        seen.push((field, normalized));
    }
    Ok(())
}

/// Whether `name` is unsafe for the given dedicated names and prefixes.
fn is_unsafe_promotion_target(name: &str, dedicated: &[&str], prefixes: &[&str]) -> bool {
    is_unsafe_promotion_target_lowercase(&name.to_ascii_lowercase(), dedicated, prefixes)
}

/// Like [`is_unsafe_promotion_target`], but `name` must already be lowercase.
fn is_unsafe_promotion_target_lowercase(name: &str, dedicated: &[&str], prefixes: &[&str]) -> bool {
    if is_transport_controlled_header_lowercase(name) || is_credential_header_lowercase(name) {
        return true;
    }
    if dedicated.iter().any(|allowed| allowed.eq_ignore_ascii_case(name)) {
        return false;
    }
    if prefixes.iter().any(|prefix| name.starts_with(prefix)) {
        return false;
    }
    name.starts_with("x-praxis-")
}

/// Reject a promotion target that would overwrite transport or routing state.
fn reject_unsafe_promotion_target(
    filter: &str,
    field: &str,
    name: Option<&str>,
    dedicated: &[&str],
    prefixes: &[&str],
) -> Result<(), FilterError> {
    let Some(raw) = name else {
        return Ok(());
    };
    let normalized = raw.to_ascii_lowercase();
    if is_unsafe_promotion_target_lowercase(&normalized, dedicated, prefixes) {
        return Err(format!(
            "{filter}: '{field}' must not use transport, credential, or internal header '{normalized}'"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn accepts_normal_model() {
        assert!(
            is_promotable_value("gpt-4.1"),
            "short ASCII model name should be promotable"
        );
    }

    #[test]
    fn rejects_oversized_value() {
        let long = "x".repeat(257);
        assert!(!is_promotable_value(&long), "257-byte value should be rejected");
    }

    #[test]
    fn accepts_value_at_limit() {
        let exact = "x".repeat(256);
        assert!(is_promotable_value(&exact), "256-byte value should be accepted");
    }

    #[test]
    fn rejects_newline() {
        assert!(!is_promotable_value("bad\nmodel"), "newline should be rejected");
    }

    #[test]
    fn accepts_empty_string() {
        assert!(is_promotable_value(""), "empty string should be accepted");
    }

    #[test]
    fn transport_controlled_headers_are_blocked() {
        for name in [
            "content-length",
            "Content-Length",
            "host",
            "transfer-encoding",
            "proxy-authorization",
            "connection",
        ] {
            assert!(
                is_transport_controlled_header(name),
                "transport header '{name}' should be blocked"
            );
        }
    }

    #[test]
    fn promotion_defaults_are_not_transport_controlled() {
        assert!(
            !is_transport_controlled_header("x-praxis-ai-effective-model"),
            "default promotion header must remain allowed"
        );
    }

    #[test]
    fn unsafe_promotion_blocks_authorization_and_unrelated_internal() {
        for name in ["authorization", "Authorization", "cookie", "x-praxis-route"] {
            assert!(
                is_unsafe_promotion_header(name),
                "promotion header '{name}' should be blocked"
            );
        }
    }

    #[test]
    fn unsafe_promotion_blocks_provider_api_key_headers() {
        for name in ["x-api-key", "X-Api-Key", "api-key", "x-goog-api-key"] {
            assert!(
                is_unsafe_promotion_header(name),
                "credential header '{name}' should be blocked"
            );
        }
    }

    #[test]
    fn unsafe_promotion_allows_ai_fact_namespaces() {
        for name in [
            "x-praxis-ai-effective-model",
            "x-praxis-ai-model",
            "x-praxis-ai-format",
            "x-praxis-responses-mode",
            "x-custom-model",
        ] {
            assert!(
                !is_unsafe_promotion_header(name),
                "promotion header '{name}' should remain allowed"
            );
        }
    }

    #[test]
    fn model_identity_promotion_blocks_cross_purpose_routing_headers() {
        for name in [
            "x-praxis-ai-format",
            "x-praxis-ai-model",
            "x-praxis-ai-stream",
            "x-praxis-responses-mode",
            "x-praxis-route",
        ] {
            assert!(
                is_unsafe_model_identity_promotion_header(name),
                "model-identity promotion must not overwrite '{name}'"
            );
        }
    }

    #[test]
    fn model_identity_promotion_allows_dedicated_and_custom_headers() {
        for name in [
            "x-praxis-ai-effective-model",
            "x-praxis-ai-original-model",
            "x-custom-model",
        ] {
            assert!(
                !is_unsafe_model_identity_promotion_header(name),
                "model-identity promotion header '{name}' should remain allowed"
            );
        }
    }

    #[test]
    fn validate_promotion_header_rejects_authorization() {
        let err = validate_promotion_header("test", "model", Some("authorization")).unwrap_err();
        assert!(
            err.to_string().contains("transport, credential, or internal header"),
            "authorization should be rejected: {err}"
        );
    }

    #[test]
    fn validate_promotion_header_rejects_x_api_key() {
        let err = validate_promotion_header("test", "model", Some("x-api-key")).unwrap_err();
        assert!(
            err.to_string().contains("x-api-key"),
            "x-api-key should be rejected: {err}"
        );
    }

    #[test]
    fn validate_model_identity_header_rejects_format_routing_fact() {
        let err = validate_model_identity_promotion_header("test", "effective_model", Some("x-praxis-ai-format"))
            .unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-format"),
            "x-praxis-ai-format should be rejected as a model-identity target: {err}"
        );
    }

    #[test]
    fn validate_model_identity_header_accepts_dedicated_default() {
        assert!(
            validate_model_identity_promotion_header("test", "effective_model", Some("x-praxis-ai-effective-model"))
                .is_ok(),
            "dedicated effective-model header should remain allowed"
        );
    }

    #[test]
    fn validate_dedicated_header_rejects_other_ai_facts() {
        let err =
            validate_dedicated_promotion_header("test", "model", Some("x-praxis-ai-format"), &["x-praxis-ai-model"])
                .unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-format"),
            "format routing header should be rejected for a model fact: {err}"
        );
    }

    #[test]
    fn validate_dedicated_header_accepts_its_own_default() {
        assert!(
            validate_dedicated_promotion_header("test", "model", Some("X-Praxis-Ai-Model"), &["x-praxis-ai-model"])
                .is_ok(),
            "dedicated model header should remain allowed case-insensitively"
        );
    }

    #[test]
    fn validate_dedicated_header_with_empty_allowlist_rejects_internal_names() {
        let err = validate_dedicated_promotion_header("test", "header", Some("x-praxis-ai-model"), &[]).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-model"),
            "empty dedicated allowlist must not overwrite AI facts: {err}"
        );
    }

    #[test]
    fn validate_dedicated_header_with_empty_allowlist_accepts_custom() {
        assert!(
            validate_dedicated_promotion_header("test", "header", Some("X-Model"), &[]).is_ok(),
            "custom non-internal header should remain allowed"
        );
    }

    #[test]
    fn credential_header_classification_matches_fixture_policy() {
        for name in [
            "authorization",
            "Authorization",
            "x-api-key",
            "api-key",
            "x-goog-api-key",
            "www-authenticate",
        ] {
            assert!(
                is_credential_header(name),
                "credential header '{name}' should be classified"
            );
        }
        assert!(
            !is_credential_header("x-custom-model"),
            "custom headers are not credentials"
        );
    }

    #[test]
    fn transport_controlled_lowercase_requires_normalized_input() {
        assert!(
            is_transport_controlled_header_lowercase("content-length"),
            "lowercase transport names must match without allocating"
        );
        assert!(
            !is_transport_controlled_header_lowercase("Content-Length"),
            "mixed-case input is the caller's responsibility"
        );
        assert!(
            is_transport_controlled_header("Content-Length"),
            "the allocating wrapper must still accept mixed case"
        );
    }

    #[test]
    fn reject_duplicate_promotion_fields_is_case_insensitive() {
        let err = reject_duplicate_promotion_fields("test", &[("format", Some("x-foo")), ("model", Some("X-Foo"))])
            .unwrap_err();
        assert!(
            err.to_string().contains("same header name"),
            "duplicate promotion fields should be rejected: {err}"
        );
    }

    #[test]
    fn reject_duplicate_promotion_fields_skips_none() {
        assert!(
            reject_duplicate_promotion_fields("test", &[("format", Some("x-foo")), ("model", None)]).is_ok(),
            "disabled fields must not collide with configured names"
        );
    }
}
