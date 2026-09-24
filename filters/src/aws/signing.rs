// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors
// Portions Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// The URI path normalization, the percent-encoding set, the excluded-header
// list and the canonical request assembly are ported from aws-sigv4 1.5.3
// and aws-smithy-http 0.64.0 (Apache-2.0), modified to compute SHA-256 and
// HMAC-SHA256 through the system OpenSSL instead of the sha2/hmac crates.

//! AWS Signature Version 4 (`SigV4`), header-based, on the system OpenSSL.
//!
//! This is the signing core of [`super::Sigv4SignFilter`]: it turns a
//! request into the `Authorization`, `x-amz-date`, `x-amz-content-sha256`
//! and (for temporary credentials) `x-amz-security-token` headers that
//! AWS services verify.
//!
//! # Why this lives here and not in `aws-sigv4`
//!
//! The official [`aws-sigv4`](https://docs.rs/aws-sigv4) crate hardwires
//! its SHA-256 and HMAC-SHA256 to the pure-Rust `sha2`/`hmac` crates, with
//! no feature or trait to substitute them, and keeps its canonical-request
//! and string-to-sign types private, so there is no seam at which only the
//! cryptography could be swapped. Praxis AI performs all of its cryptography
//! in the system OpenSSL (the validated module on a FIPS host, see
//! `docs/fips.md`), and the FIPS build rejects a binary whose dependency
//! graph names `sha2` or `hmac` at all. Signing therefore has to be
//! assembled here, over the OpenSSL-backed primitives in
//! [`praxis_ai_apis::hash`]. The protocol part (canonicalization, scope,
//! key derivation) is deliberately a line-for-line counterpart of what
//! `aws-sigv4` does for the settings this filter uses, and the test module
//! checks every output against `aws-sigv4` itself, which stays a
//! development dependency for exactly that purpose.
//!
//! # Scope
//!
//! Only what the filter needs: the signature in headers (not query
//! parameters), a fully buffered body (no `UNSIGNED-PAYLOAD`, no chunked or
//! event-stream signing), `SigV4` (not `SigV4a`), the `x-amz-content-sha256`
//! header always present, and `aws-sigv4`'s default settings otherwise:
//! URI path normalization, double percent-encoding of the path, the session
//! token included in the signed headers, and the same excluded headers.

use std::{borrow::Cow, collections::BTreeMap, fmt, str::FromStr as _, time::SystemTime};

use aws_credential_types::Credentials;
use chrono::{DateTime, Datelike as _, Timelike as _, Utc};
use http::{HeaderName, HeaderValue, Uri};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use praxis_ai_apis::hash::{HmacSha256, SHA256_LEN, Sha256, hex};
use zeroize::Zeroizing;

/// The algorithm name in the string to sign and the `Authorization` header.
const ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// Final component of the credential scope and of the key derivation.
const TERMINATOR: &str = "aws4_request";

/// Header carrying the request timestamp.
const X_AMZ_DATE: &str = "x-amz-date";

/// Header carrying the payload hash.
const X_AMZ_CONTENT_SHA256: &str = "x-amz-content-sha256";

/// Header carrying the session token of temporary credentials.
const X_AMZ_SECURITY_TOKEN: &str = "x-amz-security-token";

/// Headers never included in the signature (`aws-sigv4`'s default
/// exclusions): they are added or rewritten by intermediaries.
const EXCLUDED_HEADERS: &[&str] = &["authorization", "user-agent", "x-amzn-trace-id", "transfer-encoding"];

/// Everything but the RFC 3986 unreserved characters (`A-Z a-z 0-9 - _ . ~`),
/// percent-encoded in canonical query keys and values. Non-ASCII bytes are
/// always encoded. Mirrors the set `aws-sigv4` uses.
const QUERY_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'/')
    .add(b':')
    .add(b',')
    .add(b'?')
    .add(b'#')
    .add(b'[')
    .add(b']')
    .add(b'{')
    .add(b'}')
    .add(b'|')
    .add(b'@')
    .add(b'!')
    .add(b'$')
    .add(b'&')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*')
    .add(b'+')
    .add(b';')
    .add(b'=')
    .add(b'%')
    .add(b'<')
    .add(b'>')
    .add(b'"')
    .add(b'^')
    .add(b'`')
    .add(b'\\');

/// The query set without `/`: applied to the already percent-encoded path,
/// which `SigV4` encodes a second time for every service but S3 (`%` becomes
/// `%25`, a literal `:` becomes `%3A`, segment separators stay).
const PATH_ENCODE_SET: &AsciiSet = &QUERY_ENCODE_SET.remove(b'/');

// -----------------------------------------------------------------------------
// Inputs and errors
// -----------------------------------------------------------------------------

/// A request in the form the signature is computed over.
///
/// `uri` must be absolute (the `Host` is derived from it when no `host`
/// header is given) and already percent-encoded: `SigV4` does not encode
/// it, only re-encodes it.
pub(crate) struct SignableRequest<'a> {
    /// HTTP method, as sent.
    method: &'a str,
    /// The absolute request URI.
    uri: Uri,
    /// Headers to sign, as (name, value) pairs; names in any case.
    headers: Vec<(&'a str, &'a str)>,
    /// The complete request body.
    body: &'a [u8],
}

impl<'a> SignableRequest<'a> {
    /// Parse the URI and collect the headers.
    ///
    /// # Errors
    ///
    /// Returns [`SigningError::InvalidUri`] if `uri` does not parse, and
    /// [`SigningError::MissingAuthority`] if it has no host.
    pub(crate) fn new(
        method: &'a str,
        uri: &str,
        headers: impl Iterator<Item = (&'a str, &'a str)>,
        body: &'a [u8],
    ) -> Result<Self, SigningError> {
        let uri = Uri::from_str(uri).map_err(|e| SigningError::InvalidUri(e.to_string()))?;
        if uri.authority().is_none() {
            return Err(SigningError::MissingAuthority);
        }
        Ok(Self {
            method,
            uri,
            headers: headers.collect(),
            body,
        })
    }
}

/// The credential scope: what a signature is valid for.
#[derive(Clone, Copy)]
pub(crate) struct SigningScope<'a> {
    /// AWS region.
    pub(crate) region: &'a str,
    /// AWS signing service name.
    pub(crate) service: &'a str,
    /// Request time; the signature is valid for a window around it.
    pub(crate) time: SystemTime,
}

/// Why a request could not be signed. Never carries key material or the
/// body: only the offending signing input.
#[derive(Debug)]
pub(crate) enum SigningError {
    /// The request URI does not parse.
    InvalidUri(String),
    /// The request URI has no host to sign.
    MissingAuthority,
    /// A header name is not a valid HTTP header name: (name, reason).
    InvalidHeaderName(String, String),
    /// A header value is not a valid HTTP header value: (name, reason).
    InvalidHeaderValue(String, String),
    /// OpenSSL refused the HMAC computation (its error string).
    Mac(String),
}

impl fmt::Display for SigningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUri(e) => write!(f, "invalid request URI: {e}"),
            Self::MissingAuthority => f.write_str("request URI has no host"),
            Self::InvalidHeaderName(name, reason) => write!(f, "invalid header name '{name}': {reason}"),
            Self::InvalidHeaderValue(name, reason) => write!(f, "invalid value for header '{name}': {reason}"),
            Self::Mac(e) => write!(f, "HMAC-SHA256 failed: {e}"),
        }
    }
}

impl std::error::Error for SigningError {}

// -----------------------------------------------------------------------------
// Signing
// -----------------------------------------------------------------------------

/// Sign `request` with `credentials` for `scope`, returning the headers to
/// add to it: `x-amz-date`, `authorization`, `x-amz-content-sha256` and,
/// when the credentials carry a session token, `x-amz-security-token`.
///
/// # Errors
///
/// Returns [`SigningError`] for an unusable header or when OpenSSL refuses
/// the HMAC computation. Nothing is returned on error, so a caller can never
/// forward a partially signed request.
pub(crate) fn sign(
    request: &SignableRequest<'_>,
    credentials: &Credentials,
    scope: &SigningScope<'_>,
) -> Result<Vec<(HeaderName, HeaderValue)>, SigningError> {
    let date_time = format_date_time(scope.time);
    let payload_hash = hex(&Sha256::digest(request.body));
    let headers = canonical_headers(request, credentials.session_token(), &date_time, &payload_hash)?;
    let signed_headers = headers.keys().map(String::as_str).collect::<Vec<_>>().join(";");
    let canonical_request = canonical_request(request, &headers, &signed_headers, &payload_hash);
    let string_to_sign = string_to_sign(&date_time, scope, &hex(&Sha256::digest(canonical_request.as_bytes())));
    let signing_key = signing_key(credentials.secret_access_key(), scope)?;
    let signature = hex(&*mac(&*signing_key, string_to_sign.as_bytes())?);
    let authorization = format!(
        "{ALGORITHM} Credential={}/{}, SignedHeaders={signed_headers}, Signature={signature}",
        credentials.access_key_id(),
        credential_scope(scope)
    );
    output_headers(&date_time, &authorization, &payload_hash, credentials.session_token())
}

/// The headers to add, in the order `aws-sigv4` emits them.
fn output_headers(
    date_time: &str,
    authorization: &str,
    payload_hash: &str,
    session_token: Option<&str>,
) -> Result<Vec<(HeaderName, HeaderValue)>, SigningError> {
    let mut out = Vec::with_capacity(4);
    out.push((
        HeaderName::from_static(X_AMZ_DATE),
        header_value(X_AMZ_DATE, date_time)?,
    ));
    out.push((
        http::header::AUTHORIZATION,
        header_value("authorization", authorization)?,
    ));
    out.push((
        HeaderName::from_static(X_AMZ_CONTENT_SHA256),
        header_value(X_AMZ_CONTENT_SHA256, payload_hash)?,
    ));
    if let Some(token) = session_token {
        let mut value = header_value(X_AMZ_SECURITY_TOKEN, token)?;
        value.set_sensitive(true);
        out.push((HeaderName::from_static(X_AMZ_SECURITY_TOKEN), value));
    }
    Ok(out)
}

/// Build a header value, naming the header on failure.
fn header_value(name: &str, value: &str) -> Result<HeaderValue, SigningError> {
    HeaderValue::from_str(value).map_err(|e| SigningError::InvalidHeaderValue(name.to_owned(), e.to_string()))
}

/// Check a header name, naming it on failure.
fn check_header_name(name: &str) -> Result<(), SigningError> {
    HeaderName::from_str(name)
        .map(|_| ())
        .map_err(|e| SigningError::InvalidHeaderName(name.to_owned(), e.to_string()))
}

// -----------------------------------------------------------------------------
// Step 1: canonical request
// -----------------------------------------------------------------------------

/// The canonical headers: lowercase name to its values in request order,
/// sorted by name. Adds `host` (from the URI when the request has none),
/// `x-amz-date`, `x-amz-security-token` and `x-amz-content-sha256`, which
/// replace any same-named request header, and drops the excluded ones.
fn canonical_headers(
    request: &SignableRequest<'_>,
    session_token: Option<&str>,
    date_time: &str,
    payload_hash: &str,
) -> Result<BTreeMap<String, Vec<String>>, SigningError> {
    let mut headers: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, value) in &request.headers {
        let name = name.to_lowercase();
        check_header_name(&name)?;
        let value = trim_all(value);
        header_value(&name, &value)?;
        headers.entry(name).or_default().push(value.into_owned());
    }
    if !headers.contains_key("host") {
        headers.insert("host".to_owned(), vec![host_from_uri(&request.uri)]);
    }
    headers.insert(X_AMZ_DATE.to_owned(), vec![date_time.to_owned()]);
    if let Some(token) = session_token {
        header_value(X_AMZ_SECURITY_TOKEN, token)?;
        headers.insert(X_AMZ_SECURITY_TOKEN.to_owned(), vec![token.to_owned()]);
    }
    headers.insert(X_AMZ_CONTENT_SHA256.to_owned(), vec![payload_hash.to_owned()]);
    for excluded in EXCLUDED_HEADERS {
        headers.remove(*excluded);
    }
    Ok(headers)
}

/// The `Host` value implied by an absolute URI: the authority, without the
/// port when it is the scheme's default (RFC 9110 section 7.2), since HTTP
/// clients strip it too.
fn host_from_uri(uri: &Uri) -> String {
    let default_port = matches!(
        (uri.scheme_str(), uri.port_u16()),
        (Some("http"), Some(80)) | (Some("https"), Some(443))
    );
    if default_port {
        uri.host().unwrap_or_default().to_owned()
    } else {
        uri.authority()
            .map(http::uri::Authority::as_str)
            .unwrap_or_default()
            .to_owned()
    }
}

/// Trim leading and trailing spaces and collapse runs of spaces to one.
/// Only the space character, not other whitespace, as AWS specifies.
fn trim_all(text: &str) -> Cow<'_, str> {
    let text = text.trim_matches(' ');
    if !text.contains("  ") {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut previous_space = false;
    for c in text.chars() {
        if c == ' ' && previous_space {
            continue;
        }
        previous_space = c == ' ';
        out.push(c);
    }
    Cow::Owned(out)
}

/// The canonical request text (AWS "Create a canonical request", steps 1-7).
fn canonical_request(
    request: &SignableRequest<'_>,
    headers: &BTreeMap<String, Vec<String>>,
    signed_headers: &str,
    payload_hash: &str,
) -> String {
    let mut out = String::new();
    out.push_str(request.method);
    out.push('\n');
    out.push_str(&canonical_path(request.uri.path()));
    out.push('\n');
    out.push_str(&canonical_query(request.uri.query()));
    out.push('\n');
    for (name, values) in headers {
        out.push_str(name);
        out.push(':');
        out.push_str(&values.join(","));
        out.push('\n');
    }
    out.push('\n');
    out.push_str(signed_headers);
    out.push('\n');
    out.push_str(payload_hash);
    out
}

/// The canonical URI: the normalized path, percent-encoded once more.
fn canonical_path(path: &str) -> String {
    utf8_percent_encode(&normalize_uri_path(path), PATH_ENCODE_SET).to_string()
}

/// Resolve `.` and `..` segments and collapse empty ones, keeping a trailing
/// slash. Paths without dots or double slashes are returned as they are.
fn normalize_uri_path(path: &str) -> Cow<'_, str> {
    if path.is_empty() {
        return Cow::Borrowed("/");
    }
    let path = if path.starts_with('/') {
        Cow::Borrowed(path)
    } else {
        Cow::Owned(format!("/{path}"))
    };
    if !(path.contains('.') || path.contains("//")) {
        return path;
    }
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {},
            ".." => {
                segments.pop();
            },
            other => segments.push(other),
        }
    }
    let mut normalized = format!("/{}", segments.join("/"));
    let ends_with_slash = ["/", "/.", "/./", "/..", "/../"].iter().any(|s| path.ends_with(s));
    if ends_with_slash && !normalized.ends_with('/') {
        normalized.push('/');
    }
    Cow::Owned(normalized)
}

/// The canonical query string: every pair percent-decoded (`+` is a space),
/// re-encoded with [`QUERY_ENCODE_SET`], sorted by encoded key then value,
/// and joined as `k=v&k=v`. Empty when there is no query.
fn canonical_query(query: Option<&str>) -> String {
    let mut pairs: Vec<(String, String)> = url::form_urlencoded::parse(query.unwrap_or_default().as_bytes())
        .map(|(k, v)| {
            (
                utf8_percent_encode(&k, QUERY_ENCODE_SET).to_string(),
                utf8_percent_encode(&v, QUERY_ENCODE_SET).to_string(),
            )
        })
        .collect();
    pairs.sort();
    let mut out = String::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push('&');
        }
        out.push_str(k);
        out.push('=');
        out.push_str(v);
    }
    out
}

// -----------------------------------------------------------------------------
// Step 2: string to sign
// -----------------------------------------------------------------------------

/// The string to sign (AWS "Create a string to sign").
fn string_to_sign(date_time: &str, scope: &SigningScope<'_>, hashed_canonical_request: &str) -> String {
    format!(
        "{ALGORITHM}\n{date_time}\n{}\n{hashed_canonical_request}",
        credential_scope(scope)
    )
}

/// The credential scope `YYYYMMDD/region/service/aws4_request`.
fn credential_scope(scope: &SigningScope<'_>) -> String {
    format!(
        "{}/{}/{}/{TERMINATOR}",
        format_date(scope.time),
        scope.region,
        scope.service
    )
}

/// `YYYYMMDD` in UTC.
fn format_date(time: SystemTime) -> String {
    let time: DateTime<Utc> = time.into();
    format!("{:04}{:02}{:02}", time.year(), time.month(), time.day())
}

/// `YYYYMMDD'T'HHMMSS'Z'` in UTC.
fn format_date_time(time: SystemTime) -> String {
    let time: DateTime<Utc> = time.into();
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

// -----------------------------------------------------------------------------
// Step 3: signing key and signature
// -----------------------------------------------------------------------------

/// An HMAC-SHA256 tag that is wiped when dropped: every intermediate of the
/// key derivation is itself a key.
type Tag = Zeroizing<[u8; SHA256_LEN]>;

/// HMAC-SHA256 through the OpenSSL seam, with the error made printable
/// without naming OpenSSL types here.
fn mac(key: &[u8], data: &[u8]) -> Result<Tag, SigningError> {
    HmacSha256::mac(key, data)
        .map(Zeroizing::new)
        .map_err(|e| SigningError::Mac(e.to_string()))
}

/// Derive the signing key (AWS "Calculate the signature", step 1):
///
/// ```text
/// kDate    = HMAC("AWS4" + secret, YYYYMMDD)
/// kRegion  = HMAC(kDate, region)
/// kService = HMAC(kRegion, service)
/// kSigning = HMAC(kService, "aws4_request")
/// ```
///
/// Each key is the raw 32-byte tag of the previous step, never its hex form.
fn signing_key(secret_access_key: &str, scope: &SigningScope<'_>) -> Result<Tag, SigningError> {
    let k_secret = Zeroizing::new(format!("AWS4{secret_access_key}"));
    let k_date = mac(k_secret.as_bytes(), format_date(scope.time).as_bytes())?;
    let k_region = mac(&*k_date, scope.region.as_bytes())?;
    let k_service = mac(&*k_region, scope.service.as_bytes())?;
    mac(&*k_service, TERMINATOR.as_bytes())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::time::{Duration, SystemTime};

    use aws_credential_types::Credentials;
    use http::{HeaderName, HeaderValue};
    use praxis_ai_apis::hash::hex;

    use super::{
        SignableRequest, SigningError, SigningScope, canonical_query, format_date_time, normalize_uri_path, sign,
        signing_key, trim_all,
    };

    /// The AWS documentation example credentials.
    fn example_credentials(session_token: Option<&str>) -> Credentials {
        Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token.map(str::to_owned),
            None,
            "test",
        )
    }

    /// 2015-08-30T12:36:00Z, the timestamp of the AWS documentation example.
    fn example_time() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_440_938_160)
    }

    // -- Published vectors ------------------------------------------------

    /// AWS "Signature Version 4 signing process: examples" (IAM `ListUsers`,
    /// 20150830T123600Z, us-east-1/iam): the derived signing key and the
    /// signature over the documented string to sign. These are the bytes AWS
    /// publishes, so they pin the key derivation independently of any Rust
    /// implementation.
    #[test]
    fn signing_key_and_signature_match_the_aws_documentation_example() {
        let scope = SigningScope {
            region: "us-east-1",
            service: "iam",
            time: example_time(),
        };
        let key = signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", &scope).expect("derive");
        assert_eq!(
            hex(&*key),
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9",
            "kSigning"
        );

        let string_to_sign = "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/iam/aws4_request\n\
                              f536975d06c0309214f805bb90ccff089219ecd68b2577efef23edd43b7e1a59";
        let signature = super::mac(&*key, string_to_sign.as_bytes()).expect("mac");
        assert_eq!(
            hex(&*signature),
            "5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7",
            "signature"
        );
    }

    #[test]
    fn key_derivation_chains_raw_tags_not_hex() {
        // Re-derive step by step with the RFC 2104 primitive to show that
        // each stage keys the next with its 32 raw bytes.
        let scope = SigningScope {
            region: "us-east-1",
            service: "iam",
            time: example_time(),
        };
        let k_date = super::mac(b"AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", b"20150830").unwrap();
        let k_region = super::mac(&*k_date, b"us-east-1").unwrap();
        let k_service = super::mac(&*k_region, b"iam").unwrap();
        let k_signing = super::mac(&*k_service, b"aws4_request").unwrap();
        assert_eq!(
            *signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", &scope).unwrap(),
            *k_signing,
            "the chained derivation equals the manual one"
        );
        assert_ne!(
            *super::mac(hex(&*k_date).as_bytes(), b"us-east-1").unwrap(),
            *k_region,
            "keying the next step with hex text would give a different key"
        );
    }

    // -- Canonicalization units --------------------------------------------

    #[test]
    fn normalizes_uri_paths_like_aws() {
        for (input, expected) in [
            ("", "/"),
            ("/", "/"),
            ("/model/x/invoke", "/model/x/invoke"),
            ("no-leading-slash", "/no-leading-slash"),
            ("/a/./b", "/a/b"),
            ("/a/b/../c", "/a/c"),
            ("/a//b", "/a/b"),
            ("/a/b/", "/a/b/"),
            ("/a/b/.", "/a/b/"),
            ("/a/b/..", "/a/"),
            ("/..", "/"),
            ("/.", "/"),
        ] {
            assert_eq!(normalize_uri_path(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn canonical_path_double_encodes() {
        assert_eq!(super::canonical_path("/a%20b/c:d"), "/a%2520b/c%3Ad");
        assert_eq!(super::canonical_path("/keep-unreserved_.~"), "/keep-unreserved_.~");
    }

    #[test]
    fn canonical_query_sorts_and_encodes() {
        assert_eq!(canonical_query(None), "");
        assert_eq!(canonical_query(Some("")), "");
        assert_eq!(canonical_query(Some("b=2&a=1&a=0")), "a=0&a=1&b=2");
        assert_eq!(canonical_query(Some("k")), "k=", "a bare key has an empty value");
        assert_eq!(
            canonical_query(Some("x=a+b&y=%2F")),
            "x=a%20b&y=%2F",
            "plus is a space, slash is encoded"
        );
        assert_eq!(canonical_query(Some("=v")), "=v");
        assert_eq!(
            canonical_query(Some("Z=1&a=1")),
            "Z=1&a=1",
            "byte order, uppercase first"
        );
    }

    #[test]
    fn trims_and_collapses_spaces_only() {
        assert_eq!(trim_all("  a  b   c  "), "a b c");
        assert_eq!(trim_all("a b"), "a b");
        assert_eq!(trim_all("\ta\t"), "\ta\t", "tabs are not spaces");
    }

    #[test]
    fn formats_dates_in_utc() {
        assert_eq!(format_date_time(example_time()), "20150830T123600Z");
        assert_eq!(
            format_date_time(SystemTime::UNIX_EPOCH + Duration::from_secs(951_868_799)),
            "20000229T235959Z",
            "leap day"
        );
    }

    // -- Failure cases -------------------------------------------------------

    fn scope() -> SigningScope<'static> {
        SigningScope {
            region: "us-east-1",
            service: "bedrock",
            time: example_time(),
        }
    }

    #[test]
    fn rejects_unparseable_and_relative_uris() {
        let err = SignableRequest::new("GET", "https://h/\u{7f}", std::iter::empty(), b"").err();
        assert!(
            matches!(err, Some(SigningError::InvalidUri(_))),
            "control character: {err:?}"
        );
        let err = SignableRequest::new("GET", "/relative/only", std::iter::empty(), b"").err();
        assert!(matches!(err, Some(SigningError::MissingAuthority)), "no host: {err:?}");
    }

    #[test]
    fn rejects_invalid_header_names_and_values() {
        let request = SignableRequest::new("GET", "https://h/", [("bad name", "v")].into_iter(), b"").expect("parses");
        let err = sign(&request, &example_credentials(None), &scope()).err();
        assert!(
            matches!(&err, Some(SigningError::InvalidHeaderName(n, _)) if n == "bad name"),
            "{err:?}"
        );

        let request =
            SignableRequest::new("GET", "https://h/", [("x-ok", "bad\u{1}value")].into_iter(), b"").expect("parses");
        let err = sign(&request, &example_credentials(None), &scope()).err();
        assert!(
            matches!(&err, Some(SigningError::InvalidHeaderValue(n, _)) if n == "x-ok"),
            "{err:?}"
        );

        let request = SignableRequest::new("GET", "https://h/", std::iter::empty(), b"").expect("parses");
        let err = sign(&request, &example_credentials(Some("bad\u{1}token")), &scope()).err();
        assert!(
            matches!(&err, Some(SigningError::InvalidHeaderValue(n, _)) if n == "x-amz-security-token"),
            "{err:?}"
        );
    }

    #[test]
    fn errors_never_mention_secrets() {
        let request = SignableRequest::new("GET", "https://h/", [("bad name", "v")].into_iter(), b"").expect("parses");
        let err = sign(&request, &example_credentials(Some("SESSIONTOKENVALUE")), &scope())
            .expect_err("fails")
            .to_string();
        assert!(!err.contains("wJalrXUtnFEMI"), "no secret key in {err}");
        assert!(!err.contains("SESSIONTOKENVALUE"), "no session token in {err}");
    }

    /// With `PRAXIS_TEST_FIPS_PROVIDER` set, this test process must be on
    /// the FIPS provider (the apis crate checks the same in its own process;
    /// provider state is per process). Does nothing otherwise.
    #[test]
    fn fips_provider_is_active_when_the_run_requires_it() {
        if std::env::var_os("PRAXIS_TEST_FIPS_PROVIDER").is_none() {
            return;
        }
        praxis_tls::provider::install();
        assert!(
            praxis_tls::provider::status().provider_fips,
            "PRAXIS_TEST_FIPS_PROVIDER is set but OpenSSL does not report FIPS-approved default properties"
        );
        let scope = scope();
        let key =
            signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", &scope).expect("HMAC under the FIPS provider");
        assert_eq!(key.len(), 32, "a signing key is derived through the FIPS provider");
    }

    // -- Differential tests against aws-sigv4 -----------------------------------

    /// The signing path this module replaced: `aws-sigv4` with the settings
    /// the filter has always used. Kept verbatim as the oracle every output
    /// below is compared to.
    fn oracle(request: &Oracle<'_>) -> Result<Vec<(HeaderName, HeaderValue)>, String> {
        use aws_sigv4::{
            http_request::{PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings, sign},
            sign::v4,
        };

        let identity = request.credentials.clone().into();
        let mut signing_settings = SigningSettings::default();
        signing_settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(request.scope.region)
            .name(request.scope.service)
            .time(request.scope.time)
            .settings(signing_settings)
            .build()
            .map_err(|e| e.to_string())?
            .into();
        let signable_request = SignableRequest::new(
            request.method,
            request.uri,
            request.headers.iter().copied(),
            SignableBody::Bytes(request.body),
        )
        .map_err(|e| e.to_string())?;
        let (instructions, _signature) = sign(signable_request, &signing_params)
            .map_err(|e| e.to_string())?
            .into_parts();
        instructions.headers().map(oracle_header).collect()
    }

    /// One `aws-sigv4` signing instruction as a typed header.
    fn oracle_header((name, value): (&str, &str)) -> Result<(HeaderName, HeaderValue), String> {
        Ok((
            HeaderName::try_from(name).map_err(|e| e.to_string())?,
            HeaderValue::try_from(value).map_err(|e| e.to_string())?,
        ))
    }

    /// One differential case.
    struct Oracle<'a> {
        /// Case label for assertion messages.
        label: &'a str,
        /// HTTP method.
        method: &'a str,
        /// Absolute request URI.
        uri: &'a str,
        /// Request headers.
        headers: &'a [(&'a str, &'a str)],
        /// Request body.
        body: &'a [u8],
        /// Credentials.
        credentials: Credentials,
        /// Scope.
        scope: SigningScope<'a>,
    }

    /// Sign with this module, in the oracle's terms.
    fn ours(case: &Oracle<'_>) -> Result<Vec<(HeaderName, HeaderValue)>, SigningError> {
        let request = SignableRequest::new(case.method, case.uri, case.headers.iter().copied(), case.body)?;
        sign(&request, &case.credentials, &case.scope)
    }

    /// Compare the complete header list, name by name and byte by byte.
    fn assert_same_as_oracle(case: &Oracle<'_>) {
        let expected = oracle(case);
        let actual = ours(case);
        match (&expected, &actual) {
            (Ok(expected), Ok(actual)) => {
                let render = |headers: &[(HeaderName, HeaderValue)]| {
                    headers
                        .iter()
                        .map(|(n, v)| format!("{n}: {}", String::from_utf8_lossy(v.as_bytes())))
                        .collect::<Vec<_>>()
                };
                assert_eq!(render(actual), render(expected), "case {}", case.label);
            },
            (Err(_), Err(_)) => {},
            _ => panic!(
                "case {}: oracle {expected:?} vs ours {actual:?} disagree on success",
                case.label
            ),
        }
    }

    /// Every case: the shape of requests this filter signs plus the edges
    /// canonicalization is known to get wrong (double encoding, dots,
    /// duplicate and excluded headers, spaces, query ordering, ports,
    /// tokens, regions, services and dates).
    #[expect(clippy::too_many_lines, reason = "test data: one entry per canonicalization edge")]
    fn differential_cases() -> Vec<Oracle<'static>> {
        const HOST: (&str, &str) = ("host", "bedrock-runtime.us-east-1.amazonaws.com");
        let static_creds = || example_credentials(None);
        let temp_creds = || example_credentials(Some("FQoGZXIvYXdzEBYaDExample/Session+Token=="));
        let us_bedrock = SigningScope {
            region: "us-east-1",
            service: "bedrock",
            time: example_time(),
        };
        let eu_s3 = SigningScope {
            region: "eu-west-2",
            service: "s3",
            time: SystemTime::UNIX_EPOCH + Duration::from_secs(1_767_225_599), // 2025-12-31T23:59:59Z
        };
        let gov_api = SigningScope {
            region: "us-gov-west-1",
            service: "execute-api",
            time: SystemTime::UNIX_EPOCH + Duration::from_secs(951_868_799), // 2000-02-29T23:59:59Z
        };
        vec![
            Oracle {
                label: "bedrock invoke, static creds",
                method: "POST",
                uri: "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3/invoke",
                headers: &[HOST],
                body: br#"{"prompt":"hi"}"#,
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "bedrock invoke, temporary creds",
                method: "POST",
                uri: "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3/invoke-with-response-stream",
                headers: &[HOST],
                body: br#"{"prompt":"hi","max_tokens":1}"#,
                credentials: temp_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "arn model id, percent-encoded colons and slash (double encoding)",
                method: "POST",
                uri: "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke",
                headers: &[HOST],
                body: b"{}",
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "literal colons in the path",
                method: "POST",
                uri: "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/converse",
                headers: &[HOST],
                body: b"{}",
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "dot segments, double slashes, trailing slash",
                method: "GET",
                uri: "https://example.amazonaws.com/a/./b/../c//d/",
                headers: &[("Host", "example.amazonaws.com")],
                body: b"",
                credentials: static_creds(),
                scope: eu_s3,
            },
            Oracle {
                label: "query ordering, repeated keys, bare key, plus and percent",
                method: "GET",
                uri: "https://example.amazonaws.com/?b=2&a=1&a=0&k&x=a+b&y=%2F&Z=z&=v",
                headers: &[("host", "example.amazonaws.com")],
                body: b"",
                credentials: static_creds(),
                scope: gov_api,
            },
            Oracle {
                label: "header case, duplicates, spaces, excluded headers",
                method: "PUT",
                uri: "https://example.amazonaws.com/obj",
                headers: &[
                    ("Host", "example.amazonaws.com"),
                    ("Content-Type", "  text/plain;   charset=utf-8 "),
                    ("X-Custom", "one"),
                    ("x-custom", "two"),
                    ("User-Agent", "praxis"),
                    ("Authorization", "stale"),
                    ("X-Amzn-Trace-Id", "Root=1"),
                    ("Transfer-Encoding", "chunked"),
                    ("Range", "bytes=0-9"),
                ],
                body: b"body bytes",
                credentials: temp_creds(),
                scope: eu_s3,
            },
            Oracle {
                label: "client-supplied x-amz-date is replaced",
                method: "GET",
                uri: "https://example.amazonaws.com/",
                headers: &[("host", "example.amazonaws.com"), ("x-amz-date", "19700101T000000Z")],
                body: b"",
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "host derived from the uri, default port dropped",
                method: "GET",
                uri: "https://example.amazonaws.com:443/",
                headers: &[],
                body: b"",
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "host derived from the uri, custom port kept",
                method: "GET",
                uri: "http://127.0.0.1:3000/model/x/invoke",
                headers: &[],
                body: b"{}",
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "empty path and empty query",
                method: "DELETE",
                uri: "https://example.amazonaws.com?",
                headers: &[("host", "example.amazonaws.com")],
                body: b"",
                credentials: static_creds(),
                scope: gov_api,
            },
            Oracle {
                label: "binary body",
                method: "POST",
                uri: "https://example.amazonaws.com/upload",
                headers: &[("host", "example.amazonaws.com")],
                body: &[0, 1, 2, 3, 255, 254, 253, 10, 13, 0],
                credentials: temp_creds(),
                scope: eu_s3,
            },
            Oracle {
                label: "unparseable uri fails in both",
                method: "GET",
                uri: "https://example.amazonaws.com/model/\u{7f}/invoke",
                headers: &[],
                body: b"",
                credentials: static_creds(),
                scope: us_bedrock,
            },
            Oracle {
                label: "invalid header name fails in both",
                method: "GET",
                uri: "https://example.amazonaws.com/",
                headers: &[("host", "example.amazonaws.com"), ("bad name", "v")],
                body: b"",
                credentials: static_creds(),
                scope: us_bedrock,
            },
        ]
    }

    #[test]
    fn signs_exactly_like_aws_sigv4() {
        for case in differential_cases() {
            assert_same_as_oracle(&case);
        }
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "two scopes, each checked for key and signature")]
    fn derives_the_same_key_and_signature_as_aws_sigv4() {
        use aws_sigv4::sign::v4;
        for scope in [
            SigningScope {
                region: "us-east-1",
                service: "iam",
                time: example_time(),
            },
            SigningScope {
                region: "ap-southeast-2",
                service: "bedrock",
                time: SystemTime::UNIX_EPOCH + Duration::from_secs(1_767_225_599),
            },
        ] {
            let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
            let expected = v4::generate_signing_key(secret, scope.time, scope.region, scope.service);
            let key = signing_key(secret, &scope).unwrap();
            assert_eq!(
                &*key,
                expected.as_ref(),
                "kSigning for {}/{}",
                scope.region,
                scope.service
            );
            let string_to_sign = "AWS4-HMAC-SHA256\nstring\nto\nsign";
            assert_eq!(
                hex(&*super::mac(&*key, string_to_sign.as_bytes()).unwrap()),
                v4::calculate_signature(&expected, string_to_sign.as_bytes()),
                "signature for {}/{}",
                scope.region,
                scope.service
            );
        }
    }
}
