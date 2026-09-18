// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the `openai_conversations` example config with a
//! PostgreSQL backend that authenticates over verified TLS with a client
//! certificate (the `require_certificate_authentication` compliance
//! profile).
//!
//! The proxy connects to PostgreSQL through the platform TLS library
//! (native-tls: OpenSSL on Linux, Security.framework on macOS) and
//! presents a client certificate whose Common Name maps to the database
//! role; no password ever crosses the wire. The server's
//! `pg_hba.conf` uses only `hostssl ... cert` rules, so a connection that
//! failed to present a valid client certificate over TLS would be refused
//! before any conversation could be created.

use std::collections::HashMap;

use praxis_test_utils::{
    PostgresCertAuthGuard, example_config_path, free_port, http_send, json_post, parse_body, parse_status, patch_yaml,
    start_postgres_cert_auth, start_proxy,
};
use sqlx::{
    Row as _,
    postgres::{PgConnectOptions, PgPool, PgSslMode},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Example config exercised by these tests.
const CONFIG_PATH: &str = "openai/conversations/conversations-postgres-mtls.yaml";

/// Default conversations table name from the example config.
const CONVERSATIONS_TABLE: &str = "openai_conversations";

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires container engine (podman or docker)"]
async fn conversations_persist_over_certificate_authenticated_tls() {
    let pg = start_postgres_cert_auth();
    let proxy_port = free_port();

    let config = patched_config(&pg, proxy_port);
    let proxy = start_proxy(&config);

    // Create a conversation over the certificate-authenticated TLS
    // connection. A failure of the crypto boundary would surface here as a
    // 5xx because the store could not open its connection.
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/conversations", r#"{"metadata":{"env":"mtls"}}"#),
    );
    assert_eq!(parse_status(&raw), 200, "create conversation should return 200");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("create body should be valid JSON");
    assert_eq!(body["object"], "conversation");
    let conv_id = body["id"]
        .as_str()
        .expect("conversation id should be present")
        .to_owned();
    assert!(conv_id.starts_with("conv_"), "ID should have conv_ prefix");

    // Verify the row landed in PostgreSQL by connecting the same way the
    // proxy does: verified TLS plus a client certificate.
    let pool = verification_pool(&pg).await;
    let sql =
        format!("SELECT conversation_id, tenant_id, metadata FROM {CONVERSATIONS_TABLE} WHERE conversation_id = $1");
    let row: sqlx::postgres::PgRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind(&conv_id)
        .fetch_one(&pool)
        .await
        .expect("persisted conversation should exist in database");
    pool.close().await;

    let persisted_id: String = row.get("conversation_id");
    let tenant_id: String = row.get("tenant_id");
    let metadata_raw: String = row.get("metadata");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_raw).expect("metadata column should be valid JSON");

    assert_eq!(persisted_id, conv_id, "persisted id should match created conversation");
    assert_eq!(tenant_id, "default", "single-tenant owner should be persisted");
    assert_eq!(metadata["env"], "mtls", "persisted metadata should match");

    // A GET round-trips through the store over the same cert-auth
    // connection, proving reads work over the boundary too.
    let raw = http_send(
        proxy.addr(),
        &format!("GET /v1/conversations/{conv_id} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "GET of a persisted conversation should return 200 over cert-auth TLS"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("GET body should be valid JSON");
    assert_eq!(body["id"], conv_id, "retrieved id should match");
    assert_eq!(body["metadata"]["env"], "mtls", "retrieved metadata should match");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Patch the mTLS example config to point at the running container, using
/// freshly generated certificate material.
fn patched_config(pg: &PostgresCertAuthGuard, proxy_port: u16) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(CONFIG_PATH)).expect("example config should exist");
    let patched = patch_yaml(
        &yaml
            .replace(
                "database_url: \"postgres://praxis@db.internal:5432/praxis\"",
                &format!("database_url: \"{}\"", pg.url_without_password()),
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
        // The conversations pipeline has no upstream backend for its local
        // endpoints; the load balancer target is never dialed.
        &HashMap::new(),
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
