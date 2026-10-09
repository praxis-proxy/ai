// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the GCP ADC upstream-auth filter.

use std::{
    io::{Read as _, Write as _},
    sync::LazyLock,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use http::{HeaderValue, Method, header};
use praxis_ai_apis::subrequest::SubRequestClient;
use praxis_filter::FilterAction;
use tempfile::NamedTempFile;

use super::{
    GcpAdcFilter,
    config::{parse_gcp_adc_config, validate_service_account},
    token::{self, TokenSource, resolve_token_source},
};
use crate::test_utils::{make_filter_context, make_request};

fn test_client() -> SubRequestClient {
    crate::isolated_subrequest_client(4)
}

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

/// A one-shot HTTP/1.1 `OAuth2` token endpoint on loopback: answers any
/// request with `body` and reports the received request over `rx`, so
/// tests can inspect the signed assertion.
#[expect(
    clippy::too_many_lines,
    reason = "test fixture: header/body reassembly plus response writeback are one mock endpoint"
)]
fn mock_token_endpoint(body: &'static str) -> (String, std::sync::mpsc::Receiver<String>, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut collected = Vec::new();
        let mut buf = [0_u8; 16384];
        // Read until headers and the full Content-Length body have
        // arrived; a POST's body may land in a separate read.
        loop {
            let read = stream.read(&mut buf).unwrap();
            if read == 0 {
                break;
            }
            collected.extend_from_slice(&buf[..read]);
            let text = String::from_utf8_lossy(&collected).into_owned();
            if let Some((head, body)) = text.split_once("\r\n\r\n") {
                let expected = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                if body.len() >= expected {
                    break;
                }
            }
        }
        drop(tx.send(String::from_utf8_lossy(&collected).into_owned()));
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    (addr.to_string(), rx, handle)
}

/// `token_uri` every `key_file` test fixture points at Google's real
/// endpoint (no fetch is attempted in parse-only tests).
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// Test-only RSA material generated at runtime, never stored in the repository.
struct TestServiceAccountKey {
    /// Private key in the PKCS#8 PEM format Google's service-account key
    /// files carry.
    private_key_pem: String,
    /// Public key used to verify the JWT assertion in tests.
    public_key_pem: Vec<u8>,
}

static TEST_SERVICE_ACCOUNT_KEY: LazyLock<TestServiceAccountKey> = LazyLock::new(|| {
    let key = openssl::rsa::Rsa::generate(2048)
        .and_then(openssl::pkey::PKey::from_rsa)
        .expect("generate ephemeral test RSA key");
    TestServiceAccountKey {
        private_key_pem: String::from_utf8(key.private_key_to_pem_pkcs8().expect("serialize test private key"))
            .expect("PKCS#8 PEM is UTF-8"),
        public_key_pem: key.public_key_to_pem().expect("serialize test public key"),
    }
});

/// Email carried by every `key_file` test fixture.
const TEST_SA_EMAIL: &str = "test-sa@test-project.iam.gserviceaccount.com";

/// Write a syntactically complete `type: service_account` key file with
/// the given `token_uri` and the ephemeral test private key.
fn write_service_account_key_file(token_uri: &str) -> NamedTempFile {
    write_key_file_with_private_key(&TEST_SERVICE_ACCOUNT_KEY.private_key_pem, token_uri)
}

/// Write a `type: service_account` key file around `private_key_pem`.
fn write_key_file_with_private_key(private_key_pem: &str, token_uri: &str) -> NamedTempFile {
    // Serialized, not hand-formatted: the PEM carries newlines that
    // must arrive JSON-escaped, exactly as Google's key files do.
    let doc = serde_json::json!({
        "type": "service_account",
        "client_email": TEST_SA_EMAIL,
        "private_key": private_key_pem,
        "token_uri": token_uri,
    });
    write_json(&doc.to_string())
}

/// A PKCS#8 PEM P-256 private key: well-formed, but not RSA.
fn ec_private_key_pem() -> String {
    let group = openssl::ec::EcGroup::from_curve_name(openssl::nid::Nid::X9_62_PRIME256V1).expect("P-256");
    let pem = openssl::ec::EcKey::generate(&group)
        .and_then(openssl::pkey::PKey::from_ec_key)
        .and_then(|key| key.private_key_to_pem_pkcs8())
        .expect("generate EC test key");
    String::from_utf8(pem).expect("PEM is UTF-8")
}

/// A passphrase-protected PKCS#8 PEM RSA private key.
fn encrypted_private_key_pem() -> String {
    let pem = openssl::rsa::Rsa::generate(2048)
        .and_then(openssl::pkey::PKey::from_rsa)
        .and_then(|key| {
            key.private_key_to_pem_pkcs8_passphrase(openssl::symm::Cipher::aes_256_cbc(), b"test-passphrase")
        })
        .expect("generate encrypted test key");
    String::from_utf8(pem).expect("PEM is UTF-8")
}

/// Resolve `file` as the credentials of a `source: key_file` config.
fn resolve_key_file(file: &NamedTempFile) -> Result<TokenSource, praxis_filter::FilterError> {
    let config = parse_gcp_adc_config(&yaml(&format!(
        "source: key_file\ncredentials_file: {}",
        file.path().display()
    )))
    .expect("key_file config should parse");
    resolve_token_source(&config, None)
}

/// Build a [`TokenSource::ServiceAccountKey`] from a fixture key file.
fn service_account_key_source(token_uri: &str) -> TokenSource {
    resolve_key_file(&write_service_account_key_file(token_uri)).expect("valid key file should resolve")
}

/// Extract form field `name` from a captured token-endpoint request, or
/// `None` if the body is absent or lacks it.
fn captured_form_field(request: &str, name: &str) -> Option<String> {
    let body = request.split_once("\r\n\r\n")?.1;
    url::form_urlencoded::parse(body.as_bytes())
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// Whether the `RS256` signature on `assertion` verifies under the RSA
/// public key in `public_key_pem`.
fn assertion_signature_verifies(assertion: &str, public_key_pem: &[u8]) -> bool {
    let (signing_input, signature) = assertion.rsplit_once('.').expect("assertion must carry a signature");
    let signature = URL_SAFE_NO_PAD.decode(signature).expect("signature must be base64url");
    let public_key = openssl::pkey::PKey::public_key_from_pem(public_key_pem).expect("public PEM must parse");
    let mut verifier = openssl::sign::Verifier::new(openssl::hash::MessageDigest::sha256(), &public_key)
        .expect("SHA-256 verifier must build");
    verifier
        .set_rsa_padding(openssl::rsa::Padding::PKCS1)
        .expect("PKCS#1 padding must apply");
    verifier
        .verify_oneshot(&signature, signing_input.as_bytes())
        .expect("verification must run")
}

/// Decode the JSON in the `index`th base64url segment of `assertion`
/// (0 is the header, 1 the claims).
fn assertion_segment(assertion: &str, index: usize) -> serde_json::Value {
    let segment = assertion.split('.').nth(index).expect("assertion segment must exist");
    let json = URL_SAFE_NO_PAD.decode(segment).expect("segment must be base64url");
    serde_json::from_slice(&json).expect("segment must be JSON")
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

#[tokio::test]
async fn gcp_auth_can_be_scoped_to_selected_clusters() {
    let config = yaml("source: metadata\nclusters: [vertex, vertex-east]");
    let filter = GcpAdcFilter::from_config(&config).expect("cluster-scoped metadata config should parse");

    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(std::sync::Arc::from("anthropic"));
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "out-of-scope requests pass through"
    );
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "no GCP Authorization header is injected"
    );
}

#[test]
fn gcp_cluster_scope_rejects_blank_and_duplicate_names() {
    for yaml_text in [
        "source: metadata\nclusters: ['']",
        "source: metadata\nclusters: [vertex, vertex]",
    ] {
        assert!(
            GcpAdcFilter::from_config(&yaml(yaml_text)).is_err(),
            "must reject {yaml_text}"
        );
    }
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
    assert!(
        matches!(&source, TokenSource::Metadata { service_account } if service_account == "default"),
        "adc without env should select metadata default, got {source:?}"
    );
}

#[test]
fn adc_with_service_account_json_selects_key_file() {
    let file = write_service_account_key_file(GOOGLE_TOKEN_URL);
    let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
    let source = resolve_token_source(&config, Some(file.path())).expect("service_account JSON should select key file");
    assert!(
        matches!(
            &source,
            TokenSource::ServiceAccountKey(key)
                if key.client_email == TEST_SA_EMAIL && key.token_url == GOOGLE_TOKEN_URL
        ),
        "identity and token endpoint must come from the file, got {source:?}"
    );
}

#[test]
fn key_file_accepts_pkcs1_private_key() {
    let pem = openssl::rsa::Rsa::generate(2048)
        .and_then(|key| key.private_key_to_pem())
        .expect("generate PKCS#1 test key");
    let file = write_key_file_with_private_key(&String::from_utf8(pem).expect("PEM is UTF-8"), GOOGLE_TOKEN_URL);
    let source = resolve_key_file(&file).expect("a PKCS#1 RSA key must load");
    assert!(
        matches!(source, TokenSource::ServiceAccountKey(_)),
        "a PKCS#1 key file must select the key source, got {source:?}"
    );
}

#[test]
fn key_file_rejects_unusable_private_keys_at_config_time() {
    let ec_pem = ec_private_key_pem();
    let encrypted_pem = encrypted_private_key_pem();
    let truncated_pem = TEST_SERVICE_ACCOUNT_KEY
        .private_key_pem
        .get(..400)
        .expect("PEM is long");

    for (label, pem) in [
        ("not PEM", "not-a-key-MARKER"),
        ("truncated", truncated_pem),
        ("EC", ec_pem.as_str()),
        ("encrypted", encrypted_pem.as_str()),
    ] {
        let file = write_key_file_with_private_key(pem, GOOGLE_TOKEN_URL);
        let message = resolve_key_file(&file)
            .expect_err("an unusable private key must be rejected when the filter is built")
            .to_string();
        assert!(
            message.contains("private_key"),
            "{label}: error should name private_key, got: {message}"
        );
        let key_body = pem.lines().nth(1).unwrap_or("MARKER");
        assert!(!message.contains(key_body), "{label}: error must not echo key material");
    }
}

#[test]
fn service_account_key_debug_omits_private_key() {
    let debug = format!("{:?}", service_account_key_source(GOOGLE_TOKEN_URL));
    assert!(
        debug.contains(TEST_SA_EMAIL),
        "Debug should still identify the service account"
    );
    let leaked = TEST_SERVICE_ACCOUNT_KEY
        .private_key_pem
        .lines()
        .any(|line| debug.contains(line));
    assert!(!leaked, "Debug output must not contain private key material");
}

#[test]
fn key_file_rejects_incomplete_credentials() {
    // Every field the minting flow needs is mandatory; partial files
    // must fail at config time, not per request.
    for (label, present_fields) in [
        (
            "client_email",
            r#""private_key":"x","token_uri":"https://oauth2.googleapis.com/token""#,
        ),
        (
            "private_key",
            r#""client_email":"sa@p.iam.gserviceaccount.com","token_uri":"https://oauth2.googleapis.com/token""#,
        ),
        (
            "token_uri",
            r#""client_email":"sa@p.iam.gserviceaccount.com","private_key":"x""#,
        ),
    ] {
        let body = format!(r#"{{"type":"service_account",{present_fields}}}"#);
        let file = write_json(&body);
        let config = parse_gcp_adc_config(&yaml("{}")).expect("parse");
        let err = resolve_token_source(&config, Some(file.path()))
            .err()
            .unwrap_or_else(|| panic!("key file missing {label} must be rejected"));
        assert!(
            err.to_string().contains(label),
            "error should name the missing {label}, got: {err}"
        );
    }
}

#[test]
fn key_file_rejects_untrusted_token_uri() {
    // The token endpoint is where the signed assertion goes; only
    // Google's exact HTTPS endpoint (or a loopback test fixture) may be used.
    for token_uri in [
        "https://evil.example.com/token",
        "https://oauth2.googleapis.com.evil.test/token",
        "http://oauth2.googleapis.com/token",
        "http://169.254.169.254/token",
        "https://oauth2.googleapis.com/token?redirect=evil",
        "https://oauth2.googleapis.com/token#fragment",
        "https://user:pass@oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com:8443/token",
        "https://oauth2.googleapis.com/token/",
        "https://oauth2.googleapis.com/v4/token",
        "ftp://oauth2.googleapis.com/token",
        "http://localhost:9/token",
        "https://127.0.0.1:9/token",
        "http://user@127.0.0.1:9/token",
        "http://127.0.0.1:9/token?x=1",
        "not a url",
    ] {
        let file = write_service_account_key_file(token_uri);
        let config = parse_gcp_adc_config(&yaml(&format!(
            "source: key_file\ncredentials_file: {}",
            file.path().display()
        )))
        .expect("parse");
        let err = resolve_token_source(&config, None).expect_err("untrusted token_uri must be rejected");
        assert!(
            err.to_string().contains("token_uri"),
            "error should name token_uri for {token_uri}, got: {err}"
        );
    }
}

#[test]
fn key_file_accepts_loopback_token_uri_as_test_fixture() {
    let source = service_account_key_source("http://127.0.0.1:9/token");
    assert!(
        matches!(&source, TokenSource::ServiceAccountKey(key) if key.token_url == "http://127.0.0.1:9/token"),
        "unit-test builds must admit the loopback mock endpoint, got {source:?}"
    );
}

#[test]
fn key_file_accepts_normalized_google_token_uri() {
    for token_uri in [
        "https://oauth2.googleapis.com:443/token",
        "HTTPS://OAUTH2.GOOGLEAPIS.COM/token",
    ] {
        let source = service_account_key_source(token_uri);
        assert!(
            matches!(&source, TokenSource::ServiceAccountKey(key) if key.token_url == GOOGLE_TOKEN_URL),
            "{token_uri} is Google's endpoint and must normalize to it, got {source:?}"
        );
    }
}

#[test]
fn token_uri_errors_do_not_echo_credentials() {
    for token_uri in [
        "https://user:hunter2@oauth2.googleapis.com/token",
        "https://user:hunter2@evil.example.com/token",
        "https://user:hunter2@[bad/token",
    ] {
        let file = write_service_account_key_file(token_uri);
        let message = resolve_key_file(&file)
            .expect_err("a token_uri with credentials must be rejected")
            .to_string();
        assert!(
            message.contains("token_uri"),
            "error should name token_uri, got: {message}"
        );
        assert!(
            !message.contains("hunter2"),
            "error must not echo URL credentials, got: {message}"
        );
    }
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
    let client = test_client();
    let source = TokenSource::Metadata {
        service_account: "default".to_owned(),
    };

    let (authorization, ttl) = token::fetch(&client, &source, &host, "scope", std::time::Duration::from_secs(5))
        .await
        .expect("mock metadata fetch must succeed");

    assert_eq!(authorization.to_str().unwrap(), "Bearer abc123");
    assert!(authorization.is_sensitive(), "bearer header must be marked sensitive");
    assert_eq!(ttl, std::time::Duration::from_secs(3600));
    server.join().unwrap();
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end mint: fetch, capture, and full claim verification in one test"
)]
async fn fetch_service_account_key_mints_bearer_from_signed_assertion() {
    let (host, rx, server) = mock_token_endpoint(r#"{"access_token":"ya29.minted","expires_in":3599}"#);
    let token_url = format!("http://{host}/token");
    let source = service_account_key_source(&token_url);
    let client = test_client();

    let (authorization, ttl) = token::fetch(
        &client,
        &source,
        "unused",
        "https://www.googleapis.com/auth/cloud-platform",
        std::time::Duration::from_secs(5),
    )
    .await
    .expect("mock token endpoint mint must succeed");

    assert_eq!(authorization.to_str().unwrap(), "Bearer ya29.minted");
    assert!(authorization.is_sensitive(), "bearer header must be marked sensitive");
    assert_eq!(ttl, std::time::Duration::from_secs(3599));

    // The endpoint must have received the `JWT`-bearer grant and an
    // assertion that verifies against the fixture key's public half.
    let request = rx.recv().expect("token endpoint must record the request");
    assert!(
        request.starts_with("POST /token") && request.contains("application/x-www-form-urlencoded"),
        "mint must be a form-encoded POST, got: {request}"
    );
    assert_eq!(
        captured_form_field(&request, "grant_type").as_deref(),
        Some("urn:ietf:params:oauth:grant-type:jwt-bearer"),
        "mint must use the JWT-bearer grant"
    );
    let assertion = captured_form_field(&request, "assertion").expect("form body must carry the assertion");
    assert!(
        assertion_signature_verifies(&assertion, &TEST_SERVICE_ACCOUNT_KEY.public_key_pem),
        "assertion must verify against the fixture key's public half"
    );
    let unrelated = openssl::rsa::Rsa::generate(2048)
        .and_then(|key| key.public_key_to_pem())
        .expect("generate unrelated RSA key");
    assert!(
        !assertion_signature_verifies(&assertion, &unrelated),
        "assertion must not verify under an unrelated key"
    );
    assert_eq!(
        assertion_segment(&assertion, 0),
        serde_json::json!({"alg": "RS256", "typ": "JWT"}),
        "header must declare RS256"
    );
    let claims = assertion_segment(&assertion, 1);
    assert_eq!(claims["iss"], TEST_SA_EMAIL, "iss must be the service-account email");
    assert_eq!(
        claims["scope"], "https://www.googleapis.com/auth/cloud-platform",
        "scope must be requested"
    );
    assert_eq!(claims["aud"], token_url, "aud must be the token endpoint");
    let iat = claims["iat"].as_u64().expect("iat claim");
    let exp = claims["exp"].as_u64().expect("exp claim");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(iat.abs_diff(now) <= 60, "iat must be the signing time");
    assert_eq!(exp - iat, 600, "assertion must live for exactly ten minutes");
    server.join().unwrap();
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
async fn on_request_injects_bearer_for_key_file_source() {
    let (host, rx, server) = mock_token_endpoint(r#"{"access_token":"key-file-token","expires_in":3600}"#);
    let file = write_service_account_key_file(&format!("http://{host}/token"));
    let filter = GcpAdcFilter::from_config(&yaml(&format!(
        "source: key_file\ncredentials_file: {}",
        file.path().display()
    )))
    .expect("must construct");
    let request = make_request(Method::POST, "/v1/messages");
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
        Some("Bearer key-file-token"),
        "the minted token must be injected, not the client's credential"
    );
    assert!(rx.recv().is_ok(), "the token endpoint must have been called");
    server.join().unwrap();
}

#[tokio::test]
async fn on_request_fails_closed_for_key_file_when_token_endpoint_is_dead() {
    let file = write_service_account_key_file("http://127.0.0.1:1/token");
    let filter = GcpAdcFilter::from_config(&yaml(&format!(
        "source: key_file\ncredentials_file: {}",
        file.path().display()
    )))
    .expect("must construct");
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
    assert!(
        matches!(action, FilterAction::Reject(r) if r.status == 503),
        "a failed mint must fail closed with 503"
    );
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "no headers must be set when failing closed"
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

#[tokio::test]
async fn on_request_injects_bearer_for_in_scope_cluster() {
    let (host, server) = mock_metadata_endpoint(r#"{"access_token":"scoped","expires_in":3600}"#);
    let filter = GcpAdcFilter::from_config(&yaml(&format!(
        "source: metadata\nmetadata_host: {host}\nclusters: [vertex, vertex-east]"
    )))
    .expect("must construct");
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(std::sync::Arc::from("vertex-east"));

    let action = filter.on_request(&mut ctx).await.expect("must not error");
    assert!(matches!(action, FilterAction::Continue), "in-scope requests continue");

    let auth = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| *name == header::AUTHORIZATION)
        .map(|(_, value)| value.to_str().expect("ascii"));
    assert_eq!(
        auth,
        Some("Bearer scoped"),
        "a request routed to a listed cluster must get the token"
    );
    server.join().unwrap();
}

#[tokio::test]
async fn on_request_skips_cluster_scope_before_routing() {
    // Nothing listens on 127.0.0.1:1, so any token fetch would fail closed
    // with 503; Continue proves the filter never tried.
    let filter = GcpAdcFilter::from_config(&yaml(
        "source: metadata\nmetadata_host: 127.0.0.1:1\nclusters: [vertex]",
    ))
    .expect("must construct");
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("must not error");
    assert!(
        matches!(action, FilterAction::Continue),
        "with no cluster selected yet, a cluster-scoped filter must not fetch or reject"
    );
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "with no cluster selected yet, no GCP Authorization header is injected"
    );
}
