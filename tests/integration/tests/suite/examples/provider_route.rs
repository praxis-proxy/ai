// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the authenticated provider-route example configuration.

use std::{collections::HashMap, fmt::Write as _, path::Path, sync::Arc};

use praxis_core::config::Config;
use sha2::{Digest as _, Sha256};

const CANDIDATE_ID: &str = "inference_model/mock-model/site-us-west/provider-us-west";

/// Resolve the complete AI pipeline, including provider-boundary validation.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client =
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None));
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_ai::resolve_pipelines(&config, &registry, &health, &kv_stores, &subrequest_client)
        .map(|_pipelines| ())
        .map_err(|error| error.to_string())
}

/// Load `provider-route.yaml` and replace its mounted credential with a
/// temporary file so filter construction exercises the documented file source.
fn example_yaml() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("create credential directory");
    let token_path = dir.path().join("token");
    std::fs::write(&token_path, "provider-secret\n").expect("write provider token");

    let path = praxis_test_utils::example_config_path("provider-route.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let yaml = yaml.replace(
        "/run/secrets/provider-credentials/openai-provider/token",
        token_path.to_str().expect("temporary token path must be UTF-8"),
    );
    (dir, yaml)
}

/// Return the SHA-256 digest Praxis derives from a PEM certificate's DER bytes.
fn certificate_digest(path: &Path) -> String {
    let pem = std::fs::read(path).expect("read client certificate");
    let certificate = rustls_pemfile::certs(&mut pem.as_slice())
        .next()
        .expect("client certificate must be present")
        .expect("parse client certificate");
    let digest = Sha256::digest(certificate.as_ref());
    let mut value = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

/// Build one provider request carrying authenticated routing context.
fn provider_request(candidate: &str) -> String {
    let body = r#"{"model":"mock-model","messages":[]}"#;
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         X-AI-Routing-Candidate: {candidate}\r\n\
         X-AI-Routing-Request-Id: provider-example-test\r\n\
         Authorization: Bearer caller-secret\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    )
}

#[test]
fn provider_route_example_resolves_authenticated_pipeline() {
    let (_credentials, yaml) = example_yaml();
    resolve(&yaml).expect("provider example must satisfy the complete AI pipeline contract");
}

#[test]
fn provider_route_example_accepts_authenticated_route_and_replaces_credential() {
    let backend = praxis_test_utils::start_header_echo_backend();
    let certificates = praxis_test_utils::TestCertificates::generate();
    let client_certificate = certificates.generate_client_cert();
    let client_digest = certificate_digest(&client_certificate.cert_path);
    let proxy_port = praxis_test_utils::free_port();

    let (_credentials, yaml) = example_yaml();
    let yaml = yaml
        .replace("0.0.0.0:8443", &format!("127.0.0.1:{proxy_port}"))
        .replace(
            "/etc/praxis/tls/tls.crt",
            certificates.cert_path.to_str().expect("certificate path must be UTF-8"),
        )
        .replace(
            "/etc/praxis/tls/tls.key",
            certificates.key_path.to_str().expect("key path must be UTF-8"),
        )
        .replace(
            "/etc/praxis/tls/client-ca.crt",
            certificates
                .ca_cert_path
                .to_str()
                .expect("CA certificate path must be UTF-8"),
        )
        .replace(
            "          - organization: ai-grid",
            &format!("          - cert_digest: {client_digest}"),
        )
        .replace(
            "mock-backend.grid-demo.svc.cluster.local:8080",
            &format!("127.0.0.1:{}", backend.port()),
        );

    resolve(&yaml).expect("runtime example must satisfy provider-boundary validation");
    let config = Config::from_yaml(&yaml).expect("runtime example must parse");
    let ready_client = certificates.client_config_with_cert(&client_certificate);
    let proxy = praxis_test_utils::start_tls_proxy(&config, &ready_client);
    let request_client = certificates.raw_tls_client_config_with_cert(&client_certificate);

    let allowed = praxis_test_utils::https_send(proxy.addr(), &provider_request(CANDIDATE_ID), &request_client);
    assert_eq!(praxis_test_utils::parse_status(&allowed), 200, "{allowed}");
    let body = praxis_test_utils::parse_body(&allowed);
    assert!(
        body.contains("authorization: Bearer provider-secret"),
        "provider credential must replace caller authorization: {body}"
    );
    assert!(
        !body.contains("caller-secret"),
        "caller authorization must not reach the private backend: {body}"
    );

    let unauthenticated_client = certificates.raw_tls_client_config();
    assert!(
        praxis_test_utils::tls_connection_rejected(
            proxy.addr(),
            provider_request(CANDIDATE_ID).as_bytes(),
            &unauthenticated_client,
        ),
        "provider listener must reject clients without a certificate"
    );

    let rejected = praxis_test_utils::https_send(proxy.addr(), &provider_request("unknown-candidate"), &request_client);
    assert_eq!(
        praxis_test_utils::parse_status(&rejected),
        403,
        "unknown authenticated candidate must fail closed: {rejected}"
    );
}

#[test]
fn provider_route_example_rejects_missing_peer_trust() {
    let (_credentials, yaml) = example_yaml();
    let peer_filter = "      - filter: peer_identity_trust\n\
                       \x20       trusted_peers:\n\
                       \x20         - organization: ai-grid\n\
                       \n";
    let without_peer = yaml.replacen(peer_filter, "", 1);
    assert_ne!(without_peer, yaml, "test must remove the example peer-trust filter");

    let error = resolve(&without_peer).expect_err("provider pipeline without peer trust must fail");
    assert!(
        error.contains("requires a preceding peer_identity_trust"),
        "unexpected validation error: {error}"
    );
}
