// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Credential-store parity acceptance tests for the file-backed
//! `credential_inject` filter (praxis-proxy/praxis#1160).
//!
//! These lock the fail-closed contract against the two high-severity
//! credential-store bug classes the IPP data plane hit:
//!
//! 1. **Startup seeding** — every secret that exists before the gateway starts is usable from the first request after a
//!    (re)start, with zero secret edits. Reproduces the "empty store after restart" class.
//! 2. **Rotation under traffic** — rotating a mounted secret mid-flight flips every subsequent request to the new value
//!    with no stale/error window: each request observes either the whole old or the whole new credential, never a
//!    partial read or a 5xx. Reproduces the "stale credential under rotation" class.
//! 3. **Deletion** — deleting a secret fails that route closed with a deterministic 503, without taking down sibling
//!    routes or the chain.
//!
//! Scope: these run the real filter pipeline in-process (no cluster). "Restart"
//! is a fresh proxy start over pre-existing secret files, which exercises the
//! same construction-time seeding path as a pod restart; rotation and deletion
//! drive the file watcher through real filesystem changes under live traffic. A
//! cluster-level e2e with a projected-Secret volume and `kubectl rollout
//! restart` is tracked separately.
//!
//! All three drive the documented provider pipeline (`provider-route.yaml`):
//! `provider_route` selects a candidate and emits its credential secretRef,
//! then `credential_inject` resolves the mounted file and replaces the caller
//! `Authorization` before the backend hop. The header-echo backend reflects the
//! injected `Authorization` so the assertions observe exactly which credential
//! reached the upstream.

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use praxis_core::config::Config;
use sha2::{Digest as _, Sha256};

/// Namespace shared by the demo secretRefs, matching `provider-route.yaml`.
const NAMESPACE: &str = "grid-demo";

/// Deadline for a rotation or deletion to converge through the file watcher.
const CONVERGE_TIMEOUT: Duration = Duration::from_secs(5);

/// Interval between convergence polls.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// One configured credential: its route selector and mounted token file.
struct CredentialFixture {
    /// Edge-selected candidate id (`x-ai-routing-candidate`).
    candidate: String,
    /// Model accepted for this candidate.
    model: String,
    /// Kubernetes Secret name in the secretRef.
    name: String,
    /// Mounted token file path.
    path: PathBuf,
    /// Current token bytes written to `path`.
    token: String,
}

impl CredentialFixture {
    /// Build fixture `index` in `dir`, writing its initial token to disk.
    fn seed(dir: &Path, index: usize, token: &str) -> Self {
        let name = format!("provider-secret-{index}");
        let path = dir.join(format!("token-{index}"));
        std::fs::write(&path, format!("{token}\n")).expect("write mounted token");
        Self {
            candidate: format!("inference_model/model-{index}/site-us-west/provider-us-west"),
            model: format!("mock-model-{index}"),
            name,
            path,
            token: token.to_owned(),
        }
    }

    /// Atomically replace the mounted token, mirroring a projected-volume swap.
    fn rotate(&mut self, token: &str) {
        let staged = self.path.with_extension("next");
        std::fs::write(&staged, format!("{token}\n")).expect("stage rotated token");
        std::fs::rename(&staged, &self.path).expect("atomically swap rotated token");
        token.clone_into(&mut self.token);
    }
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

/// Render a provider pipeline config with one route + credential per fixture,
/// mirroring the `provider-route.yaml` example structure.
fn build_yaml(
    proxy_port: u16,
    certificates: &praxis_test_utils::TestCertificates,
    client_digest: &str,
    backend_port: u16,
    fixtures: &[CredentialFixture],
) -> String {
    let cert_path = certificates.cert_path.to_str().expect("cert path must be UTF-8");
    let key_path = certificates.key_path.to_str().expect("key path must be UTF-8");
    let ca_path = certificates.ca_cert_path.to_str().expect("CA path must be UTF-8");

    let mut routes = String::new();
    let mut credentials = String::new();
    for fixture in fixtures {
        let file = fixture.path.to_str().expect("token path must be UTF-8");
        write!(
            routes,
            "          - candidate_id: {candidate}\n\
             \x20           model: {model}\n\
             \x20           paths:\n\
             \x20             - /v1/chat/completions\n\
             \x20             - /v1/responses\n\
             \x20           cluster: mock-backend\n\
             \x20           credential:\n\
             \x20             strategy: bearer_token\n\
             \x20             secretRef:\n\
             \x20               name: {name}\n\
             \x20               namespace: {NAMESPACE}\n\
             \x20               key: token\n",
            candidate = fixture.candidate,
            model = fixture.model,
            name = fixture.name,
        )
        .expect("writing to String cannot fail");
        write!(
            credentials,
            "          - strategy: bearer_token\n\
             \x20           name: {name}\n\
             \x20           namespace: {NAMESPACE}\n\
             \x20           key: token\n\
             \x20           file: {file}\n",
            name = fixture.name,
        )
        .expect("writing to String cannot fail");
    }

    format!(
        "listeners:\n\
         \x20 - name: provider\n\
         \x20   address: \"127.0.0.1:{proxy_port}\"\n\
         \x20   filter_chains:\n\
         \x20     - provider-inference\n\
         \x20   tls:\n\
         \x20     certificates:\n\
         \x20       - cert_path: {cert_path}\n\
         \x20         key_path: {key_path}\n\
         \x20     client_ca:\n\
         \x20       ca_path: {ca_path}\n\
         \x20     client_cert_mode: require\n\
         filter_chains:\n\
         \x20 - name: provider-inference\n\
         \x20   filters:\n\
         \x20     - filter: peer_identity_trust\n\
         \x20       trusted_peers:\n\
         \x20         - cert_digest: {client_digest}\n\
         \x20     - filter: json_body_field\n\
         \x20       field: model\n\
         \x20       header: X-Model\n\
         \x20     - filter: provider_route\n\
         \x20       provider_id: site-us-west\n\
         \x20       model_header: X-Model\n\
         \x20       emit_demo_attribution: true\n\
         \x20       routes:\n\
         {routes}\
         \x20     - filter: credential_inject\n\
         \x20       credentials:\n\
         {credentials}\
         \x20     - filter: load_balancer\n\
         \x20       clusters:\n\
         \x20         - name: mock-backend\n\
         \x20           endpoints:\n\
         \x20             - \"127.0.0.1:{backend_port}\"\n\
         admin:\n\
         \x20 address: \"127.0.0.1:0\"\n\
         insecure_options:\n\
         \x20 allow_private_endpoints: true\n\
         shutdown_timeout_secs: 5\n",
    )
}

/// Build one authenticated provider request for the given candidate + model.
fn provider_request(candidate: &str, model: &str) -> String {
    let body = format!(r#"{{"model":"{model}","messages":[]}}"#);
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         X-AI-Routing-Candidate: {candidate}\r\n\
         X-AI-Routing-Request-Id: credential-parity-test\r\n\
         Authorization: Bearer caller-secret\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len(),
    )
}

/// mTLS scaffolding shared by every scenario: certificates, a header-echo
/// backend, and the authenticated client identity trusted by the pipeline.
struct Harness {
    certificates: praxis_test_utils::TestCertificates,
    client_certificate: praxis_test_utils::ClientCert,
    client_digest: String,
    backend: praxis_test_utils::BackendGuard,
}

impl Harness {
    fn new() -> Self {
        let certificates = praxis_test_utils::TestCertificates::generate();
        let client_certificate = certificates.generate_client_cert();
        let client_digest = certificate_digest(&client_certificate.cert_path);
        Self {
            certificates,
            client_certificate,
            client_digest,
            backend: praxis_test_utils::start_header_echo_backend(),
        }
    }

    /// Start a fresh proxy instance for the given fixtures on a new port.
    ///
    /// A fresh start rebuilds the pipeline, which resolves and seeds every
    /// mounted credential at construction — the same path a pod restart takes.
    fn start(&self, fixtures: &[CredentialFixture]) -> praxis_test_utils::ProxyGuard {
        let yaml = build_yaml(
            praxis_test_utils::free_port(),
            &self.certificates,
            &self.client_digest,
            self.backend.port(),
            fixtures,
        );
        let config = Config::from_yaml(&yaml).expect("generated provider config must parse");
        let ready_client = self.certificates.client_config_with_cert(&self.client_certificate);
        praxis_test_utils::start_tls_proxy(&config, &ready_client)
    }

    /// Send one authenticated request over the mTLS client connection.
    fn send(&self, proxy: &praxis_test_utils::ProxyGuard, fixture: &CredentialFixture) -> String {
        let request = provider_request(&fixture.candidate, &fixture.model);
        let client = self
            .certificates
            .raw_tls_client_config_with_cert(&self.client_certificate);
        praxis_test_utils::https_send(proxy.addr(), &request, &client)
    }
}

/// The header-echo backend lower-cases header names but preserves the value.
fn injected_authorization(body: &str, token: &str) -> bool {
    body.contains(&format!("authorization: Bearer {token}"))
}

/// Assert a fixture routes successfully and injects exactly its own credential.
fn assert_injects_own_credential(
    harness: &Harness,
    proxy: &praxis_test_utils::ProxyGuard,
    fixture: &CredentialFixture,
) {
    let response = harness.send(proxy, fixture);
    assert_eq!(
        praxis_test_utils::parse_status(&response),
        200,
        "credential '{}' must route: {response}",
        fixture.name
    );
    let body = praxis_test_utils::parse_body(&response);
    assert!(
        injected_authorization(&body, &fixture.token),
        "credential '{}' must be injected: {body}",
        fixture.name
    );
    assert!(
        !body.contains("caller-secret"),
        "caller authorization must never reach the backend: {body}"
    );
}

#[test]
fn credential_store_seeds_every_secret_across_a_restart() {
    // Startup-seeding class: secrets created before the gateway started must be
    // usable from the first request after a (re)start, with zero edits.
    let harness = Harness::new();
    let dir = tempfile::tempdir().expect("create credential directory");
    let fixtures: Vec<CredentialFixture> = (0..4)
        .map(|index| CredentialFixture::seed(dir.path(), index, &format!("seed-token-{index}")))
        .collect();

    // First boot: every pre-existing secret is immediately usable.
    let proxy = harness.start(&fixtures);
    for fixture in &fixtures {
        assert_injects_own_credential(&harness, &proxy, fixture);
    }

    // Restart with the same on-disk secrets and no edits: the store must
    // re-seed completely, so the first post-restart request already works.
    drop(proxy);
    let restarted = harness.start(&fixtures);
    for fixture in &fixtures {
        assert_injects_own_credential(&harness, &restarted, fixture);
    }
}

#[test]
fn credential_rotation_under_traffic_flips_without_stale_or_error_window() {
    // Stale-credential-under-rotation class: once the mounted secret rotates,
    // every subsequent request must use the new value, and no request may
    // observe a partial read or a 5xx during the transition.
    let harness = Harness::new();
    let dir = tempfile::tempdir().expect("create credential directory");
    let mut fixture = CredentialFixture::seed(dir.path(), 0, "rotation-token-a");
    let old_token = fixture.token.clone();

    let proxy = harness.start(std::slice::from_ref(&fixture));

    let before = harness.send(&proxy, &fixture);
    assert_eq!(praxis_test_utils::parse_status(&before), 200, "{before}");
    assert!(
        injected_authorization(&praxis_test_utils::parse_body(&before), &old_token),
        "pre-rotation traffic must use the original credential: {before}"
    );

    let new_token = "rotation-token-b";
    fixture.rotate(new_token);

    // Poll under continuous traffic until the new value is observed. Every
    // response must stay 200 and carry either the old or the new credential.
    let deadline = Instant::now() + CONVERGE_TIMEOUT;
    let mut converged = false;
    while Instant::now() < deadline {
        let response = harness.send(&proxy, &fixture);
        assert_eq!(
            praxis_test_utils::parse_status(&response),
            200,
            "rotation must not open an error window: {response}"
        );
        let body = praxis_test_utils::parse_body(&response);
        assert!(
            injected_authorization(&body, &old_token) || injected_authorization(&body, new_token),
            "each request must carry the old or new credential, never a partial read: {body}"
        );
        if injected_authorization(&body, new_token) {
            converged = true;
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    assert!(converged, "rotated credential was never observed within the deadline");

    // From the rotation point onward the new value is stable.
    for _ in 0..4 {
        let response = harness.send(&proxy, &fixture);
        assert_eq!(praxis_test_utils::parse_status(&response), 200, "{response}");
        let body = praxis_test_utils::parse_body(&response);
        assert!(
            injected_authorization(&body, new_token),
            "post-rotation traffic must be stable on the new credential: {body}"
        );
        assert!(
            !injected_authorization(&body, &old_token),
            "the old credential must not resurface after convergence: {body}"
        );
    }
}

#[test]
fn credential_deletion_fails_route_closed_without_taking_down_siblings() {
    // Deletion contract: deleting a secret fails its route closed with a
    // deterministic 503, while sibling routes stay healthy and the filter
    // chain does not crash.
    let harness = Harness::new();
    let dir = tempfile::tempdir().expect("create credential directory");
    let fixtures = vec![
        CredentialFixture::seed(dir.path(), 0, "survivor-token"),
        CredentialFixture::seed(dir.path(), 1, "doomed-token"),
    ];
    let survivor = &fixtures[0];
    let doomed = &fixtures[1];

    let proxy = harness.start(&fixtures);

    // Both routes work before deletion.
    for fixture in &fixtures {
        assert_injects_own_credential(&harness, &proxy, fixture);
    }

    std::fs::remove_file(&doomed.path).expect("delete mounted secret");

    // The deleted route converges to a deterministic 503 fail-closed. Until it
    // does, it may only serve its still-valid credential (never an error).
    let deadline = Instant::now() + CONVERGE_TIMEOUT;
    let mut failed_closed = false;
    while Instant::now() < deadline {
        let response = harness.send(&proxy, doomed);
        let status = praxis_test_utils::parse_status(&response);
        if status == 503 {
            let body = praxis_test_utils::parse_body(&response);
            assert!(
                !body.contains("doomed-token"),
                "fail-closed response must not leak the credential: {body}"
            );
            failed_closed = true;
            break;
        }
        assert_eq!(
            status, 200,
            "before failing closed the route may only serve the still-valid credential: {response}"
        );
        std::thread::sleep(POLL_INTERVAL);
    }
    assert!(failed_closed, "deleted credential route never failed closed");

    // Deletion is stable and does not resurrect a stale value.
    for _ in 0..3 {
        let response = harness.send(&proxy, doomed);
        assert_eq!(
            praxis_test_utils::parse_status(&response),
            503,
            "deleted credential must stay failed closed: {response}"
        );
    }

    // The sibling route is unaffected by the deletion and reload.
    assert_injects_own_credential(&harness, &proxy, survivor);
}
