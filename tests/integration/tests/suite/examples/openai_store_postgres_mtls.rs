// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the `openai_store` example config with a
//! PostgreSQL backend that authenticates over verified TLS with a client
//! certificate (the `require_certificate_authentication` compliance
//! profile).
//!
//! These tests exercise the cryptographic boundary end to end: the proxy
//! connects to PostgreSQL through the platform TLS library (native-tls:
//! OpenSSL on Linux, Security.framework on macOS), presents a client
//! certificate whose Common Name maps to the database role, and persists a
//! response without any password ever crossing the wire. The
//! server's `pg_hba.conf` uses only `hostssl ... cert` rules, so a
//! connection that failed to present a valid client certificate over TLS
//! would be refused before any row could be written.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, PostgresCertAuthGuard, example_config_path, free_port, http_send, json_post, parse_body, parse_status,
    patch_yaml, start_postgres_cert_auth, start_proxy,
};
use sqlx::{
    Row as _,
    postgres::{PgConnectOptions, PgPool, PgSslMode},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Backend response matching a real Responses API shape with `input`
/// and `output` fields the store extracts for persistence.
const RESPONSE_JSON: &str = r#"{"id":"resp_mtls_abc","created_at":3000,"model":"gpt-4.1","object":"response","input":"Hello over mTLS","output":[{"type":"message","content":[{"type":"output_text","text":"Hi there over mTLS"}]}]}"#;

/// Example config exercised by these tests.
const CONFIG_PATH: &str = "openai/responses/response-store-postgres-mtls.yaml";

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires container engine (podman or docker)"]
async fn response_store_persists_over_certificate_authenticated_tls() {
    let pg = start_postgres_cert_auth();

    let backend_guard = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let suffix = unique_suffix();
    let responses_table = format!("openai_responses_{suffix}");
    let conversations_table = format!("openai_conversations_{suffix}");

    let config = patched_config(
        &pg,
        proxy_port,
        backend_guard.port(),
        &responses_table,
        &conversations_table,
    );
    let proxy = start_proxy(&config);

    // POST persists the response over the certificate-authenticated TLS
    // connection. A failure of the crypto boundary would surface here as a
    // 5xx because the store could not open its connection.
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello over mTLS"}"#),
    );
    assert_eq!(parse_status(&raw), 200, "Responses API POST should return 200");
    assert_eq!(
        parse_body(&raw),
        RESPONSE_JSON,
        "response body should match the backend's JSON"
    );

    // Verify the row landed in PostgreSQL by connecting the same way the
    // proxy does: verified TLS plus a client certificate. This directly
    // exercises SQLx's TLS cryptographic boundary from the test as well.
    let pool = verification_pool(&pg).await;
    let sql = format!("SELECT id, tenant_id, created_at, model FROM {responses_table} WHERE id = $1");
    let row: sqlx::postgres::PgRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind("resp_mtls_abc")
        .fetch_one(&pool)
        .await
        .expect("persisted record should exist in database");
    pool.close().await;

    let id: String = row.get("id");
    let tenant_id: String = row.get("tenant_id");
    let created_at: i64 = row.get("created_at");
    let model: String = row.get("model");

    assert_eq!(id, "resp_mtls_abc", "persisted id should match response");
    assert_eq!(tenant_id, "default", "single-tenant owner should be persisted");
    assert_eq!(created_at, 3000, "persisted created_at should match response");
    assert_eq!(model, "gpt-4.1", "persisted model should match response");

    // A GET round-trips through the store over the same cert-auth
    // connection, proving reads work over the boundary too.
    let raw = http_send(
        proxy.addr(),
        "GET /v1/responses/resp_mtls_abc HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "GET of a persisted response should return 200 over cert-auth TLS"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("GET body should be valid JSON");
    assert_eq!(body["id"], "resp_mtls_abc", "retrieved id should match");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Patch the mTLS example config to point at the running container and
/// backend, using freshly generated certificate material and unique table
/// names for parallel isolation.
fn patched_config(
    pg: &PostgresCertAuthGuard,
    proxy_port: u16,
    backend_port: u16,
    responses_table: &str,
    conversations_table: &str,
) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(CONFIG_PATH)).expect("example config should exist");
    let patched = patch_yaml(
        &yaml
            .replace(
                "database_url: \"postgres://praxis@db.internal:5432/praxis\"",
                &format!("database_url: \"{}\"", pg.url_without_password()),
            )
            .replace(
                "        responses_table: openai_responses",
                &format!("        responses_table: {responses_table}"),
            )
            .replace(
                "        conversations_table: openai_conversations",
                &format!("        conversations_table: {conversations_table}"),
            )
            .replace(
                "ssl_root_cert: /etc/praxis/pki/ca.crt",
                &format!("ssl_root_cert: {}", pg.ca_cert_path().display()),
            )
            .replace(
                "ssl_client_cert: /etc/praxis/pki/client.crt",
                &format!("ssl_client_cert: {}", pg.client_cert_path().display()),
            )
            .replace(
                "ssl_client_key: /etc/praxis/pki/client.key",
                &format!("ssl_client_key: {}", pg.client_key_path().display()),
            ),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_port)]),
    );
    praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse")
}

/// Build a verification connection pool that authenticates the same way
/// the proxy does: verified TLS with the client certificate.
async fn verification_pool(pg: &PostgresCertAuthGuard) -> PgPool {
    let options: PgConnectOptions = pg
        .url_without_password()
        .parse::<PgConnectOptions>()
        .expect("password-less URL should parse")
        .ssl_mode(PgSslMode::VerifyFull)
        .ssl_root_cert(pg.ca_cert_path())
        .ssl_client_cert(pg.client_cert_path())
        .ssl_client_key(pg.client_key_path());
    Box::pin(PgPool::connect_with(options))
        .await
        .expect("verification connection should authenticate over cert-auth TLS")
}

/// Generate a unique suffix for table names to allow parallel test
/// execution against a shared PostgreSQL instance.
fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = std::thread::current().id();
    format!("{id}_{tid:?}")
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_")
        .to_lowercase()
}
