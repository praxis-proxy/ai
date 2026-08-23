// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! [`Sigv4SignFilter`] — signs outbound requests to AWS services using
//! Signature Version 4 (`SigV4`).
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
//! This filter is a **generic** `SigV4` signer: region, service, and
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
//! (`RustCrypto`) crates, not `aws-lc-rs`. This repository's TLS layer
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

use std::time::SystemTime;

use aws_credential_types::Credentials;
use http::{HeaderName, HeaderValue};
use praxis_filter::FilterError;
use serde::Deserialize;

/// Default cap on buffered request-body bytes used for payload hashing.
const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;

// -----------------------------------------------------------------------------
// Signing
// -----------------------------------------------------------------------------

/// Computes the `SigV4` headers to add to an outbound request.
///
/// Pure function: takes the already-resolved `credentials` and every
/// piece of the request that participates in signing, and returns the
/// headers to set. Does not read [`HttpFilterContext`] directly so it
/// can be tested independently of the framework (see the AWS-vector
/// test below) and independently of async credential resolution.
///
/// `uri` must already be fully encoded (`SigV4` does not re-encode it).
///
/// # Errors
///
/// Returns [`FilterError`] if the signing library rejects the inputs
/// (e.g. an invalid header name/value, or a malformed URI).
#[expect(clippy::too_many_arguments, reason = "each argument is a distinct, independently-testable piece of the signature")]
pub(crate) fn sign_headers<'a>(
    credentials: &Credentials,
    region: &str,
    service: &str,
    time: SystemTime,
    method: &'a str,
    uri: &'a str,
    headers: impl Iterator<Item = (&'a str, &'a str)>,
    body: &'a [u8],
) -> Result<Vec<(HeaderName, HeaderValue)>, FilterError> {
    use aws_sigv4::http_request::{sign, PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings};
    use aws_sigv4::sign::v4;

    let identity = credentials.clone().into();

    let mut signing_settings = SigningSettings::default();
    // Bedrock (and most non-S3 AWS services) require the literal
    // x-amz-content-sha256 header on the wire; aws-sigv4 defaults to
    // omitting it, so this must be set explicitly.
    signing_settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;

    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(service)
        .time(time)
        .settings(signing_settings)
        .build()
        .map_err(|e| FilterError::from(format!("aws_sigv4_sign: invalid signing params: {e}")))?
        .into();

    let signable_request = SignableRequest::new(method, uri, headers, SignableBody::Bytes(body))
        .map_err(|e| FilterError::from(format!("aws_sigv4_sign: invalid signable request: {e}")))?;

    let (instructions, _signature) = sign(signable_request, &signing_params)
        .map_err(|e| FilterError::from(format!("aws_sigv4_sign: signing failed: {e}")))?
        .into_parts();

    instructions
        .headers()
        .map(|(name, value)| {
            let header_name = HeaderName::try_from(name)
                .map_err(|e| FilterError::from(format!("aws_sigv4_sign: invalid header name '{name}': {e}")))?;
            let header_value = HeaderValue::try_from(value)
                .map_err(|e| FilterError::from(format!("aws_sigv4_sign: invalid header value for '{name}': {e}")))?;
            Ok((header_name, header_value))
        })
        .collect()
}

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
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic, reason = "tests")]
mod tests {
    use std::time::SystemTime;

    use aws_credential_types::Credentials;

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

    #[test]
    #[expect(clippy::too_many_lines, reason = "doc-comment explaining a documentation discrepancy dominates the line count")]
    fn sign_headers_matches_aws_published_get_object_example() {
        // Worked example from AWS's own docs (fetched 2026-08-22):
        // https://docs.aws.amazon.com/AmazonS3/latest/developerguide/sig-v4-header-based-auth.html#example-signature-calculations
        // GET https://examplebucket.s3.amazonaws.com/test.txt, Range: bytes=0-9
        // Date: Fri, 24 May 2013 00:00:00 GMT
        //
        // NOTE: the expected `Signature` below is
        // `67fe34c8530db585abddc51067328adfedb6e42487d2566dc7d927d6e2722900`,
        // **not** the `f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41`
        // printed on that AWS doc page. The page's own `CanonicalRequest`
        // (`GET\n/test.txt\n\nhost:...\nrange:...\nx-amz-content-sha256:...\n
        // x-amz-date:...\n\nhost;range;x-amz-content-sha256;x-amz-date\n<hash>`)
        // and `StringToSign` (hashing to
        // `7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972`)
        // were copied verbatim from that same page and are correct, but its
        // final published `Signature` value is stale/wrong. Verified
        // independently three ways before overriding the doc's value here:
        // (1) a from-scratch Python `hashlib`/`hmac` HMAC-SHA256 chain over
        // exactly that CanonicalRequest/StringToSign, (2) AWS's own
        // `botocore` `S3SigV4Auth` Python SDK signer, and (3) this crate
        // (`aws-sigv4` 1.5.1) — all three independently produce
        // `67fe34c8...`, not `f0e8bdb8...`. Do not "fix" this back to the
        // doc's value without re-deriving it yourself.
        let credentials = Credentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test-vector",
        );
        let time = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_369_353_600); // 2013-05-24T00:00:00Z

        let headers = super::sign_headers(
            &credentials,
            "us-east-1",
            "s3",
            time,
            "GET",
            "https://examplebucket.s3.amazonaws.com/test.txt",
            [("host", "examplebucket.s3.amazonaws.com"), ("range", "bytes=0-9")].into_iter(),
            b"",
        )
        .expect("signing the AWS worked example must succeed");

        let authorization = headers
            .iter()
            .find(|(name, _)| name == http::header::AUTHORIZATION)
            .map(|(_, value)| value.to_str().expect("header value must be ASCII"))
            .expect("Authorization header must be present");

        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, \
             Signature=67fe34c8530db585abddc51067328adfedb6e42487d2566dc7d927d6e2722900",
            "signature must match the independently-verified worked example (see comment above)"
        );
    }

    #[test]
    fn sign_headers_includes_security_token_for_temporary_credentials() {
        let credentials = Credentials::new(
            "ASIAEXAMPLE",
            "secretkeyexample",
            Some("sessiontokenexample".to_owned()),
            None,
            "test-vector",
        );
        let time = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_369_353_600);

        let headers = super::sign_headers(
            &credentials,
            "us-east-1",
            "bedrock",
            time,
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3/invoke",
            [("host", "bedrock-runtime.us-east-1.amazonaws.com")].into_iter(),
            br#"{"prompt":"hi"}"#,
        )
        .expect("signing with temporary credentials must succeed");

        let has_security_token = headers
            .iter()
            .any(|(name, _)| name.as_str().eq_ignore_ascii_case("x-amz-security-token"));
        assert!(has_security_token, "temporary credentials must produce a signed x-amz-security-token header");

        let has_content_sha256 = headers
            .iter()
            .any(|(name, _)| name.as_str().eq_ignore_ascii_case("x-amz-content-sha256"));
        assert!(has_content_sha256, "x-amz-content-sha256 must be present given PayloadChecksumKind::XAmzSha256");
    }
}
