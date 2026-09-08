// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the GCP ADC upstream-auth filter.

use std::io::{Read as _, Write as _};

use http::{HeaderValue, Method, header};
use praxis_filter::FilterAction;
use tempfile::NamedTempFile;

use super::{
    GcpAdcFilter,
    config::{parse_gcp_adc_config, validate_service_account},
    token::{self, TokenSource, resolve_token_source},
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

/// Spawn a one-shot HTTP/1.1 server on loopback that replies with `body`
/// to any request, and returns its bound `host:port`.
fn mock_metadata_endpoint(body: &'static str) -> (String, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 4096];
        let _ = stream.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    (addr.to_string(), handle)
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
    assert_eq!(
        config.metadata_host, "metadata.google.internal",
        "metadata_host should default"
    );
}

#[test]
fn parses_explicit_metadata_and_key_file() {
    let config = parse_gcp_adc_config(&yaml(
        "
source: metadata
service_account: foo@project.iam.gserviceaccount.com
scope: https://www.googleapis.com/auth/cloud-platform
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
fn rejects_structural_characters_in_metadata_host() {
    let err = GcpAdcFilter::from_config(&yaml("metadata_host: evil.com/../x"))
        .err()
        .expect("path-injecting metadata_host must be rejected");
    assert!(
        err.to_string().contains("metadata_host"),
        "error should name metadata_host: {err}"
    );
}

#[test]
fn rejects_non_loopback_non_default_metadata_host() {
    // Structurally valid hostname, but not the real metadata server or a
    // loopback test address -- must still be rejected, or a misconfigured
    // metadata_host would send the access token over a real network in
    // cleartext.
    let err = GcpAdcFilter::from_config(&yaml("metadata_host: evil.example.com"))
        .err()
        .expect("non-loopback, non-default metadata_host must be rejected");
    assert!(
        err.to_string().contains("metadata_host"),
        "error should name metadata_host: {err}"
    );
}

#[test]
fn accepts_loopback_and_default_metadata_host() {
    GcpAdcFilter::from_config(&yaml("source: metadata\nmetadata_host: metadata.google.internal"))
        .expect("the real metadata server must be accepted");
    GcpAdcFilter::from_config(&yaml("source: metadata\nmetadata_host: 127.0.0.1:9000"))
        .expect("a loopback IP address must be accepted for tests");
}

#[test]
fn rejects_localhost_metadata_host() {
    // Unlike a literal loopback IP, `localhost` is a hostname resolved
    // via DNS/`/etc/hosts` and could be remapped to point anywhere --
    // accepting it would defeat the loopback restriction entirely.
    let err = GcpAdcFilter::from_config(&yaml("metadata_host: localhost:9000"))
        .err()
        .expect("localhost must be rejected, it is not a fixed address");
    assert!(
        err.to_string().contains("metadata_host"),
        "error should name metadata_host: {err}"
    );
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
// token::fetch against a mock metadata endpoint
// -----------------------------------------------------------------------------

#[tokio::test]
async fn fetch_parses_bearer_and_ttl_for_metadata_source() {
    let (host, server) = mock_metadata_endpoint(r#"{"access_token":"abc123","expires_in":3600}"#);
    let client = reqwest::Client::new();
    let source = TokenSource::Metadata {
        service_account: "default".to_owned(),
    };

    let (authorization, ttl) = token::fetch(&client, &source, &host, "scope")
        .await
        .expect("mock metadata fetch must succeed");

    assert_eq!(authorization.to_str().unwrap(), "Bearer abc123");
    assert!(authorization.is_sensitive(), "bearer header must be marked sensitive");
    assert_eq!(ttl, std::time::Duration::from_secs(3600));
    server.join().unwrap();
}

#[tokio::test]
async fn fetch_errors_for_service_account_key_source() {
    let client = reqwest::Client::new();
    let err = token::fetch(&client, &TokenSource::ServiceAccountKey, "unused", "scope")
        .await
        .expect_err("key_file fetch is not implemented and must error, not hang or silently fail closed forever");
    assert!(
        err.to_string().contains("not implemented"),
        "error must explain why, got: {err}"
    );
}

// -----------------------------------------------------------------------------
// on_request: cache-through end to end
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_injects_bearer_on_first_fetch() {
    let (host, server) = mock_metadata_endpoint(r#"{"access_token":"fresh","expires_in":3600}"#);
    let filter =
        GcpAdcFilter::from_config(&yaml(&format!("source: metadata\nmetadata_host: {host}"))).expect("must construct");
    let request = make_request(Method::POST, "/v1/models");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must not error");
    assert!(matches!(action, FilterAction::Continue));

    let auth = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| *name == header::AUTHORIZATION)
        .map(|(_, value)| value.to_str().expect("ascii"));
    assert_eq!(
        auth,
        Some("Bearer fresh"),
        "must inject the freshly fetched bearer token"
    );
    server.join().unwrap();
}

#[tokio::test]
async fn on_request_reuses_cached_token_without_a_second_fetch() {
    // The mock endpoint accepts exactly one connection; a second
    // on_request call must not attempt a second fetch, or it would fail
    // to connect and 503 instead of continuing.
    let (host, server) = mock_metadata_endpoint(r#"{"access_token":"once","expires_in":3600}"#);
    let filter =
        GcpAdcFilter::from_config(&yaml(&format!("source: metadata\nmetadata_host: {host}"))).expect("must construct");
    let request = make_request(Method::POST, "/v1/models");

    let mut first_ctx = make_filter_context(&request);
    let first = filter
        .on_request(&mut first_ctx)
        .await
        .expect("first call must not error");
    assert!(matches!(first, FilterAction::Continue));

    let mut second_ctx = make_filter_context(&request);
    let second = filter
        .on_request(&mut second_ctx)
        .await
        .expect("second call must not error");
    assert!(
        matches!(second, FilterAction::Continue),
        "a still-valid cache must serve the second request without a new connection"
    );
    let auth = second_ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| *name == header::AUTHORIZATION)
        .map(|(_, value)| value.to_str().expect("ascii"));
    assert_eq!(auth, Some("Bearer once"));
    server.join().unwrap();
}

#[tokio::test]
async fn on_request_fails_closed_when_metadata_unreachable() {
    let filter =
        GcpAdcFilter::from_config(&yaml("source: metadata\nmetadata_host: 127.0.0.1:1")).expect("must construct");
    let request = make_request(Method::POST, "/v1/models");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
    assert!(
        matches!(action, FilterAction::Reject(r) if r.status == 503),
        "a failed fetch must fail closed with 503"
    );
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "no headers must be set when failing closed"
    );
}

#[tokio::test]
async fn on_request_fails_closed_for_key_file_source() {
    let file = write_json(r#"{"type":"service_account","client_email":"sa@example.com"}"#);
    let filter = GcpAdcFilter::from_config(&yaml(&format!(
        "source: key_file\ncredentials_file: {}",
        file.path().display()
    )))
    .expect("must construct");
    let request = make_request(Method::POST, "/v1/models");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
    assert!(
        matches!(action, FilterAction::Reject(r) if r.status == 503),
        "key_file token fetch is not implemented yet and must fail closed with 503"
    );
}

#[tokio::test]
async fn on_request_overwrites_client_authorization() {
    let (host, server) = mock_metadata_endpoint(r#"{"access_token":"gcp-token","expires_in":3600}"#);
    let filter =
        GcpAdcFilter::from_config(&yaml(&format!("source: metadata\nmetadata_host: {host}"))).expect("must construct");
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
    server.join().unwrap();
}
