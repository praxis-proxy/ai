// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the GCP ADC upstream-auth filter.

use std::{
    io::Write as _,
    time::{Duration, Instant},
};

use http::{HeaderValue, Method, header};
use praxis_filter::{FilterAction, HttpFilter as _};
use tempfile::NamedTempFile;

use super::{
    GcpAdcFilter,
    config::{parse_gcp_adc_config, validate_service_account},
    filter::CachedToken,
    token::{TokenSource, resolve_token_source},
};
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn yaml(body: &str) -> serde_yaml::Value {
    serde_yaml::from_str(body).expect("test YAML must parse")
}

fn write_json(body: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("temp file");
    file.write_all(body.as_bytes()).expect("write json");
    file.flush().expect("flush json");
    file
}

fn valid_token(access_token: &str) -> CachedToken {
    CachedToken::new(access_token, Instant::now() + Duration::from_secs(300)).expect("valid test token")
}

// -----------------------------------------------------------------------------
// Config parsing
// -----------------------------------------------------------------------------

#[test]
fn parses_minimal_valid_config() {
    let config = parse_gcp_adc_config(&yaml("{}")).expect("empty config should parse with defaults");

    assert_eq!(config.source, super::config::GcpAdcSource::Adc);
    assert_eq!(config.scope, "https://www.googleapis.com/auth/cloud-platform");
    assert!(config.service_account.is_none(), "service_account should be unset");
    assert!(config.credentials_file.is_none(), "credentials_file should be unset");
    assert!(
        (config.refresh_ratio - 0.75).abs() < f64::EPSILON,
        "default refresh_ratio should be 0.75"
    );
}

#[test]
fn parses_explicit_metadata_and_key_file() {
    let config = parse_gcp_adc_config(&yaml(
        "
source: metadata
service_account: foo@project.iam.gserviceaccount.com
scope: https://www.googleapis.com/auth/cloud-platform
refresh_ratio: 0.5
",
    ))
    .expect("metadata config should parse");
    assert_eq!(config.source, super::config::GcpAdcSource::Metadata);
    assert_eq!(
        config.service_account.as_deref(),
        Some("foo@project.iam.gserviceaccount.com")
    );

    let config = parse_gcp_adc_config(&yaml(
        "
source: key_file
credentials_file: /var/secrets/sa.json
",
    ))
    .expect("key_file config should parse");
    assert_eq!(config.source, super::config::GcpAdcSource::KeyFile);
    assert_eq!(config.credentials_file.as_deref(), Some("/var/secrets/sa.json"));
}

#[test]
fn rejects_missing_credentials_file_for_key_file() {
    let err = GcpAdcFilter::from_config(&yaml("source: key_file"))
        .err()
        .expect("key_file without path must fail");
    assert!(
        err.to_string().contains("credentials_file"),
        "error should name credentials_file: {err}"
    );
}

#[test]
fn rejects_credentials_file_for_non_key_file_sources() {
    for source in ["adc", "metadata"] {
        let err = GcpAdcFilter::from_config(&yaml(&format!(
            "source: {source}\ncredentials_file: /var/secrets/sa.json"
        )))
        .err()
        .expect("credentials_file must be rejected when the source does not use it");
        assert!(
            err.to_string().contains("credentials_file"),
            "error should name credentials_file for source {source}: {err}"
        );
    }
}

#[test]
fn rejects_service_account_for_key_file_source() {
    let err = GcpAdcFilter::from_config(&yaml(
        "source: key_file\n\
         credentials_file: /var/secrets/sa.json\n\
         service_account: foo@project.iam.gserviceaccount.com",
    ))
    .err()
    .expect("service_account must be rejected when the source does not use it");
    assert!(
        err.to_string().contains("service_account"),
        "error should name service_account: {err}"
    );
}

#[test]
fn rejects_unknown_field() {
    let err = parse_gcp_adc_config(&yaml("audience: https://example.com")).expect_err("audience must be unknown");
    assert!(
        err.to_string().contains("audience"),
        "unknown field error should mention audience: {err}"
    );
}

#[test]
fn from_config_rejects_out_of_range_refresh_ratio() {
    for ratio in ["0", "1", "-0.1", "1.5"] {
        let err = GcpAdcFilter::from_config(&yaml(&format!("refresh_ratio: {ratio}")))
            .err()
            .expect("refresh_ratio must be exclusive (0, 1)");
        assert!(
            err.to_string().contains("refresh_ratio"),
            "error should name refresh_ratio for {ratio}: {err}"
        );
    }
}

#[test]
fn validate_service_account_accepts_email_and_default() {
    validate_service_account("default").expect("default is valid");
    validate_service_account("foo@project.iam.gserviceaccount.com").expect("SA email is valid");
}

#[test]
fn validate_service_account_rejects_unsafe_values() {
    // Allowlist: anything outside letters/digits/@.-_ must be rejected,
    // including percent-encoding that could smuggle path structure into
    // the metadata URL.
    for value in [
        "",
        "default/../token",
        "a?b",
        "sa#frag",
        "sa token",
        "sa%2Ftoken",
        "sa:8080",
    ] {
        validate_service_account(value).expect_err("must reject structurally unsafe service_account");
    }
}

#[test]
fn from_config_builds_filter_for_metadata_source() {
    let filter = GcpAdcFilter::from_config(&yaml("source: metadata")).expect("metadata config should construct");
    assert_eq!(filter.name(), "gcp_adc");
}

// -----------------------------------------------------------------------------
// ADC selection
// -----------------------------------------------------------------------------

#[test]
fn adc_without_env_selects_metadata() {
    let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
    let source = resolve_token_source(&config, None).expect("adc without env should select metadata");
    assert_eq!(
        source,
        TokenSource::Metadata {
            service_account: "default".to_owned(),
        },
        "adc without env should select metadata default"
    );
}

#[test]
fn adc_with_service_account_json_selects_key_file() {
    let file = write_json(r#"{"type":"service_account","client_email":"sa@example.com"}"#);
    let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
    let source = resolve_token_source(&config, Some(file.path())).expect("service_account JSON should select key file");
    assert!(
        matches!(source, TokenSource::ServiceAccountKey),
        "expected service account key source, got {source:?}"
    );
}

#[test]
fn adc_rejects_authorized_user() {
    let file = write_json(r#"{"type":"authorized_user"}"#);
    let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
    let err = resolve_token_source(&config, Some(file.path())).expect_err("authorized_user must be rejected");
    assert!(
        err.to_string().contains("authorized_user"),
        "error should mention authorized_user: {err}"
    );
}

#[test]
fn adc_rejects_external_account() {
    let file = write_json(r#"{"type":"external_account"}"#);
    let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
    let err = resolve_token_source(&config, Some(file.path())).expect_err("external_account must be rejected");
    assert!(
        err.to_string().contains("external_account"),
        "error should mention external_account: {err}"
    );
}

#[test]
fn adc_rejects_missing_credentials_file() {
    let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
    let err = resolve_token_source(&config, Some(std::path::Path::new("/no/such/gcp-adc-credentials.json")))
        .expect_err("missing ADC file must fail");
    assert!(err.to_string().contains("gcp_adc"), "error should be namespaced: {err}");
}

#[test]
fn explicit_metadata_ignores_credentials_path() {
    let file = write_json(r#"{"type":"authorized_user"}"#);
    let config = parse_gcp_adc_config(&yaml("source: metadata")).expect("parse");
    let source = resolve_token_source(&config, Some(file.path())).expect("metadata must ignore ADC file");
    assert!(
        matches!(source, TokenSource::Metadata { .. }),
        "explicit metadata must not read GOOGLE_APPLICATION_CREDENTIALS, got {source:?}"
    );
}

// -----------------------------------------------------------------------------
// CachedToken
// -----------------------------------------------------------------------------

#[tokio::test]
async fn cached_token_formats_bearer_and_marks_sensitive() {
    let request = make_request(Method::GET, "/");
    let filter = GcpAdcFilter::for_test(Some(valid_token("secret-token")));
    let mut ctx = make_filter_context(&request);

    // The constructor — not the test — must produce a sensitive,
    // Bearer-formatted header value.
    let action = filter.on_request(&mut ctx).await.expect("must not error");
    assert!(matches!(action, FilterAction::Continue));
    let injected = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| *name == header::AUTHORIZATION)
        .map(|(_, value)| value);
    let injected = injected.expect("Authorization must be injected");
    assert_eq!(injected.to_str().expect("ascii"), "Bearer secret-token");
    assert!(injected.is_sensitive(), "constructor must mark the value sensitive");
}

#[test]
fn cached_token_rejects_invalid_header_value() {
    CachedToken::new("bad\ntoken", Instant::now()).expect_err("control characters must be rejected");
}

#[test]
fn cached_token_expiry() {
    let now = Instant::now();
    let token = CachedToken::new("t", now + Duration::from_secs(60)).expect("valid token");
    assert!(token.is_valid(now), "token in the future must be valid");
    assert!(
        !token.is_valid(now + Duration::from_secs(61)),
        "token past expiry must be invalid"
    );
}

// -----------------------------------------------------------------------------
// on_request
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_injects_bearer_when_token_valid() {
    let filter = GcpAdcFilter::for_test(Some(valid_token("test-token")));
    let request = make_request(Method::POST, "/v1/models");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must not error");
    assert!(matches!(action, FilterAction::Continue), "valid token must continue");

    let auth = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| *name == header::AUTHORIZATION)
        .map(|(_, value)| value.to_str().expect("ascii"));
    assert_eq!(auth, Some("Bearer test-token"), "must inject the cached bearer token");
}

#[tokio::test]
async fn on_request_fails_closed_when_no_token() {
    let filter = GcpAdcFilter::for_test(None);
    let request = make_request(Method::POST, "/v1/models");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
    assert!(
        matches!(action, FilterAction::Reject(rejection) if rejection.status == 503),
        "no cached token must fail closed with 503"
    );
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "no headers must be set when failing closed"
    );
}

#[tokio::test]
async fn on_request_fails_closed_when_token_expired() {
    let filter = GcpAdcFilter::for_test(Some(
        CachedToken::new("stale", Instant::now() - Duration::from_secs(1)).expect("valid header value"),
    ));
    let request = make_request(Method::POST, "/v1/models");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
    assert!(
        matches!(action, FilterAction::Reject(rejection) if rejection.status == 503),
        "expired token must fail closed with 503"
    );
}

#[tokio::test]
async fn on_request_overwrites_client_authorization() {
    let filter = GcpAdcFilter::for_test(Some(valid_token("gcp-token")));
    let mut request = make_request(Method::POST, "/v1/models");
    request
        .headers
        .insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer client"));
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must not error");
    assert!(matches!(action, FilterAction::Continue));

    let auth = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| *name == header::AUTHORIZATION)
        .map(|(_, value)| value.to_str().expect("ascii"));
    assert_eq!(auth, Some("Bearer gcp-token"));
}
