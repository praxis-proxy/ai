// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! [`Sigv4SignFilter`] — signs outbound requests to AWS services using
//! Signature Version 4 (SigV4).
//!
//! # Overview
//!
//! AWS services (Bedrock and others) require every request to be
//! cryptographically signed: a canonical form of the method, path,
//! headers, and a SHA-256 hash of the body is combined with an AWS
//! secret key via HMAC-SHA256 to produce an `Authorization` header
//! that proves the caller holds the key and that the request was not
//! tampered with in transit.
//!
//! This filter is a **generic** SigV4 signer: region, service, and
//! target host are all configuration, not hardcoded. It does not know
//! about Bedrock specifically, and it does not translate request
//! bodies between provider formats — that belongs in separate filters.
//!
//! # Ordering requirement
//!
//! This filter reads [`HttpFilterContext::rewritten_path`] and sets
//! its own `Host` header. It must be placed **after** any
//! `path_rewrite`/`url_rewrite` filter in the same chain, and it owns
//! the `Host` header for the request — do not pair it with another
//! filter (such as `headers`) that also sets `Host`.
//!
//! By the time this filter runs, `ctx.request` is still the
//! **client's original** request; the actual upstream path is
//! [`HttpFilterContext::rewritten_path`] when set (applied to the wire
//! later, in the protocol layer's `upstream_request_filter`), and
//! there is no framework-computed "final upstream Host" at all. This
//! filter sets `Host` itself; the operator must point the cluster's
//! `endpoints` and `tls.sni` at that same host.
//!
//! # Credentials
//!
//! This filter currently supports **static credentials only** (access
//! key, secret key, optional session token — each read from an
//! environment variable at filter construction time). OIDC / web
//! identity federation and the AWS default credential chain are a
//! planned follow-up that will plug into the same
//! [`aws_credential_types::provider::SharedCredentialsProvider`] seam
//! this filter already uses internally, with no change to the signing
//! or request-handling code.
//!
//! # FIPS considerations
//!
//! Signing uses the [`aws-sigv4`](https://docs.rs/aws-sigv4) crate,
//! which computes HMAC-SHA256 via the pure-Rust `hmac`/`sha2`
//! (RustCrypto) crates, not `aws-lc-rs`. This repository's TLS layer
//! uses `aws-lc-rs` (which has a FIPS-140-3-validated build mode), but
//! that does not extend to this filter's signing computation, which
//! goes through a different, non-FIPS-validated code path. There is
//! currently no `aws-lc-rs`-backed alternative to `aws-sigv4` upstream.
//! If FIPS-validated request signing becomes a hard requirement, this
//! filter will need revisiting; it is not addressed here.
//!
//! # YAML config
//!
//! ```yaml
//! filter: aws_sigv4_sign
//! region: us-east-1
//! service: bedrock
//! host: bedrock-runtime.us-east-1.amazonaws.com
//! access_key_env_var: AWS_ACCESS_KEY_ID
//! secret_key_env_var: AWS_SECRET_ACCESS_KEY
//! session_token_env_var: AWS_SESSION_TOKEN   # optional
//! max_body_bytes: 1048576                    # optional, default 1 MiB
//! ```

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default cap on buffered request-body bytes used for payload hashing.
const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `aws_sigv4_sign` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Sigv4SignConfig {
    /// AWS region used in the signing scope (e.g. `us-east-1`).
    pub(crate) region: String,

    /// AWS signing service name (e.g. `bedrock`). Note this is the
    /// *signing* name, which can differ from the hostname — Bedrock's
    /// runtime host is `bedrock-runtime.*` but its signing service
    /// name is `bedrock`.
    pub(crate) service: String,

    /// Host header to sign and to set on the outbound request. Must
    /// match the host the operator has configured for the target
    /// cluster's `endpoints`/`tls.sni`.
    pub(crate) host: String,

    /// Environment variable holding the AWS access key ID.
    pub(crate) access_key_env_var: String,

    /// Environment variable holding the AWS secret access key.
    pub(crate) secret_key_env_var: String,

    /// Environment variable holding an optional AWS session token
    /// (required for temporary/STS-issued credentials).
    #[serde(default)]
    pub(crate) session_token_env_var: Option<String>,

    /// Maximum buffered request-body bytes for payload hashing.
    /// Requests with a larger body are rejected with 413 by the
    /// framework before this filter runs.
    #[serde(default = "default_max_body_bytes")]
    pub(crate) max_body_bytes: usize,
}

/// Default value for [`Sigv4SignConfig::max_body_bytes`].
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Parse and validate the `aws_sigv4_sign` filter's YAML config.
///
/// # Errors
///
/// Returns [`FilterError`] if the YAML is malformed, has unknown
/// fields, or is missing a required field.
pub(crate) fn parse_sigv4_config(config: &serde_yaml::Value) -> Result<Sigv4SignConfig, FilterError> {
    praxis_filter::parse_filter_config("aws_sigv4_sign", config)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{DEFAULT_MAX_BODY_BYTES, parse_sigv4_config};

    fn yaml(body: &str) -> serde_yaml::Value {
        serde_yaml::from_str(body).expect("test YAML must parse")
    }

    #[test]
    fn parses_minimal_valid_config() {
        let config = parse_sigv4_config(&yaml(
            "region: us-east-1\n\
             service: bedrock\n\
             host: bedrock-runtime.us-east-1.amazonaws.com\n\
             access_key_env_var: AWS_ACCESS_KEY_ID\n\
             secret_key_env_var: AWS_SECRET_ACCESS_KEY\n",
        ))
        .expect("minimal config should parse");

        assert_eq!(config.region, "us-east-1");
        assert_eq!(config.service, "bedrock");
        assert_eq!(config.host, "bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(config.access_key_env_var, "AWS_ACCESS_KEY_ID");
        assert_eq!(config.secret_key_env_var, "AWS_SECRET_ACCESS_KEY");
        assert!(config.session_token_env_var.is_none(), "session token should default to None");
        assert_eq!(config.max_body_bytes, DEFAULT_MAX_BODY_BYTES, "should apply default max_body_bytes");
    }

    #[test]
    fn parses_optional_session_token_and_max_body_bytes() {
        let config = parse_sigv4_config(&yaml(
            "region: us-east-1\n\
             service: bedrock\n\
             host: bedrock-runtime.us-east-1.amazonaws.com\n\
             access_key_env_var: AWS_ACCESS_KEY_ID\n\
             secret_key_env_var: AWS_SECRET_ACCESS_KEY\n\
             session_token_env_var: AWS_SESSION_TOKEN\n\
             max_body_bytes: 2048\n",
        ))
        .expect("full config should parse");

        assert_eq!(config.session_token_env_var.as_deref(), Some("AWS_SESSION_TOKEN"));
        assert_eq!(config.max_body_bytes, 2048);
    }

    #[test]
    fn rejects_missing_required_field() {
        let err = parse_sigv4_config(&yaml(
            "service: bedrock\n\
             host: bedrock-runtime.us-east-1.amazonaws.com\n\
             access_key_env_var: AWS_ACCESS_KEY_ID\n\
             secret_key_env_var: AWS_SECRET_ACCESS_KEY\n",
        ));
        assert!(err.is_err(), "missing 'region' should fail to parse");
    }

    #[test]
    fn rejects_unknown_field() {
        let err = parse_sigv4_config(&yaml(
            "region: us-east-1\n\
             service: bedrock\n\
             host: bedrock-runtime.us-east-1.amazonaws.com\n\
             access_key_env_var: AWS_ACCESS_KEY_ID\n\
             secret_key_env_var: AWS_SECRET_ACCESS_KEY\n\
             typo_field: oops\n",
        ));
        assert!(err.is_err(), "unknown field should be rejected by deny_unknown_fields");
    }
}
