// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::collections::BTreeMap;

use bytes::Bytes;
use http::Method;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, HttpFilter, parse_filter_config};
use serde_json::Value;

use super::{
    config::{ConversationsConfig, revalidate_postgres_host, validate_config},
    filter::OpenaiConversationsFilter,
    routes::{self, ConversationOperation, ConversationOperationSpec, operation_specs},
    validate::validate_metadata,
};
use crate::{
    openai::responses::state::ResponsesState,
    test_utils::{make_filter_context, make_request, make_response},
};

fn rejection_body(rejection: &praxis_filter::Rejection) -> Value {
    serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap()
}

// -----------------------------------------------------------------------------
// Config Tests
// -----------------------------------------------------------------------------

#[test]
fn parse_valid_sqlite_config() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn parse_valid_postgres_config() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/conversations"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn reject_empty_database_url() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: ""
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("must not be empty"),
        "expected empty URL error: {err}"
    );
}

#[test]
fn reject_duplicate_table_names() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: same_name
        items_table: same_name
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("distinct"),
        "expected distinct table names error: {err}"
    );
}

#[test]
fn reject_items_table_matching_generated_responses_table() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversations_unused_responses
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("generated responses and items table names"),
        "expected generated response table collision error: {err}"
    );
}

#[test]
fn reject_invalid_table_name() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: "1invalid"
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("invalid conversations_table"),
        "expected invalid table name error: {err}"
    );
}

#[test]
fn reject_postgres_items_table_above_index_safe_length() {
    let items_table = "i".repeat(64);
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/conversations"
        conversations_table: conversations
        items_table: {items_table}
        "#
    ))
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("items table name"),
        "expected postgres items table length error: {err}"
    );
}

#[test]
fn reject_sqlite_path_traversal() {
    for database_url in [
        "sqlite://../../etc/data.db",
        "sqlite://..%2F..%2Fetc%2Fdata.db?mode=rwc",
    ] {
        let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
            r#"
            backend: sqlite
            database_url: "{database_url}"
            conversations_table: conversations
            items_table: conversation_items
            "#
        ))
        .unwrap();
        let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("path traversal"),
            "expected path traversal error for {database_url}: {err}"
        );
    }
}

#[test]
fn reject_ssl_mode_on_sqlite() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        ssl_mode: require
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("only valid with the 'postgres' backend"),
        "expected postgres-only error: {err}"
    );
}

#[test]
fn reject_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        unknown_field: true
        "#,
    )
    .unwrap();
    let result = parse_filter_config::<ConversationsConfig>("openai_conversations", &yaml);
    assert!(result.is_err(), "should reject unknown fields");
}

#[test]
fn reject_postgres_without_scheme() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "1.2.3.4:5432/conversations"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("must start with"),
        "expected scheme error: {err}"
    );
}

// -----------------------------------------------------------------------------
// Config Tests — Postgres URL Validation
// -----------------------------------------------------------------------------

#[test]
fn reject_postgres_loopback_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://127.0.0.1:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "loopback IP should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_private_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://192.168.1.1:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "private IP should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_link_local_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://169.254.1.1:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "link-local IP should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_unspecified_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://0.0.0.0:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "unspecified IP should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_ipv6_loopback() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://[::1]:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "IPv6 loopback should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_ipv6_unique_local() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://[fd00::1]:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "IPv6 unique-local should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_ipv6_link_local() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://[fe80::1]:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "IPv6 link-local should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_ipv6_unspecified() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://[::]:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "IPv6 unspecified should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_dns_name() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://db.example.com:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("DNS name"),
        "DNS name should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_localhost() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://localhost:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("localhost"),
        "localhost should be rejected: {err}"
    );
}

#[test]
fn allow_private_database_url_bypasses_ip_checks() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://127.0.0.1:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        allow_private_database_url: true
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn allow_private_database_url_bypasses_dns_checks() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://db.example.com:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        allow_private_database_url: true
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn reject_postgres_unix_socket() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres:///db?host=/var/run/postgresql"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("Unix socket"),
        "Unix socket should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_no_explicit_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres:///db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("explicit host"),
        "missing host should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_hostaddr_private() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db?hostaddr=127.0.0.1"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "private hostaddr should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_host_query_param_localhost() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres:///db?host=localhost"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("localhost"),
        "localhost host param should be rejected: {err}"
    );
}

#[test]
fn accept_postgresql_scheme() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgresql://1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn postgres_url_with_credentials_validates_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://user:pass@1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn reject_postgres_mapped_ipv4_loopback() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://[::ffff:127.0.0.1]:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "IPv4-mapped IPv6 loopback should be rejected: {err}"
    );
}

// -----------------------------------------------------------------------------
// Config Tests — Postgres TLS
// -----------------------------------------------------------------------------

#[test]
fn reject_ssl_root_cert_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        ssl_mode: verify-ca
        ssl_root_cert: "../../etc/ca.pem"
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("path traversal"),
        "ssl_root_cert path traversal should be rejected: {err}"
    );
}

#[test]
fn reject_ssl_root_cert_without_verify_mode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        ssl_mode: require
        ssl_root_cert: "/path/to/ca.pem"
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("verify-ca"),
        "ssl_root_cert without verify mode should be rejected: {err}"
    );
}

#[test]
fn accept_ssl_root_cert_with_verify_ca() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        ssl_mode: verify-ca
        ssl_root_cert: "/path/to/ca.pem"
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn accept_ssl_root_cert_with_verify_full() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        ssl_mode: verify-full
        ssl_root_cert: "/path/to/ca.pem"
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn reject_postgres_url_tls_file_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db?sslrootcert=../../etc/ca.pem"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("path traversal"),
        "sslrootcert path traversal should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_url_sslkey_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db?sslkey=../../etc/key.pem"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("path traversal"),
        "sslkey path traversal should be rejected: {err}"
    );
}

#[test]
fn url_sslmode_verify_ca_with_sslrootcert_is_valid() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db?sslmode=verify-ca&sslrootcert=/ca.pem"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

// -----------------------------------------------------------------------------
// Config Tests — SQLite Extras
// -----------------------------------------------------------------------------

#[test]
fn reject_ssl_root_cert_on_sqlite() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        ssl_root_cert: "/path/to/ca.pem"
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("only valid with the 'postgres' backend"),
        "ssl_root_cert on sqlite should be rejected: {err}"
    );
}

#[test]
fn reject_allow_private_on_sqlite() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        allow_private_database_url: true
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("only valid with the 'postgres' backend"),
        "allow_private_database_url on sqlite should be rejected: {err}"
    );
}

#[test]
fn accept_sqlite_memory_mode_query_param() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite://file?mode=memory"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn accept_sqlite_colon_memory_variant() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite://:memory:"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn accept_sqlite_file_path_without_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite://data/conversations.db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
}

#[test]
fn default_table_names_are_valid() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    validate_config(&cfg).unwrap();
    assert_eq!(cfg.conversations_table, "openai_conversations");
    assert_eq!(cfg.items_table, "openai_conversation_items");
}

#[test]
fn reject_postgres_conversations_table_above_index_safe_length() {
    let table = "c".repeat(64);
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/conversations"
        conversations_table: {table}
        items_table: conversation_items
        "#
    ))
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("conversations_table") || err.to_string().contains("table"),
        "expected postgres table length error: {err}"
    );
}

// -----------------------------------------------------------------------------
// Config Tests — Legacy IPv4 Parsing
// -----------------------------------------------------------------------------

#[test]
fn reject_postgres_octal_loopback() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://0177.0.0.01:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "octal 127.0.0.1 should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_hex_loopback() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://0x7f000001:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "hex 127.0.0.1 should be rejected: {err}"
    );
}

#[test]
fn reject_postgres_decimal_collapsed_loopback() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://2130706433:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = validate_config(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "decimal 127.0.0.1 (2130706433) should be rejected: {err}"
    );
}

// -----------------------------------------------------------------------------
// Config Tests — revalidate_postgres_host
// -----------------------------------------------------------------------------

#[test]
fn revalidate_postgres_host_rejects_private_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://10.0.0.1:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = revalidate_postgres_host(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "revalidation should reject private IP: {err}"
    );
}

#[test]
fn revalidate_postgres_host_rejects_hostaddr_param() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db?hostaddr=192.168.0.1"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    let err = revalidate_postgres_host(&cfg).unwrap_err();
    assert!(
        err.to_string().contains("local-sensitive"),
        "revalidation should reject private hostaddr: {err}"
    );
}

#[test]
fn revalidate_postgres_host_accepts_public_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: postgres
        database_url: "postgres://1.2.3.4:5432/db"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    revalidate_postgres_host(&cfg).unwrap();
}

#[test]
fn revalidate_skips_sqlite_backend() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let cfg: ConversationsConfig = parse_filter_config("openai_conversations", &yaml).unwrap();
    revalidate_postgres_host(&cfg).unwrap();
}

// -----------------------------------------------------------------------------
// Metadata Validation Tests
// -----------------------------------------------------------------------------

#[test]
fn valid_metadata() {
    let metadata = serde_json::json!({"key1": "value1", "key2": "value2"});
    validate_metadata(&metadata).unwrap();
}

#[test]
fn null_metadata_is_valid() {
    validate_metadata(&Value::Null).unwrap();
}

#[test]
fn reject_non_object_metadata() {
    let metadata = serde_json::json!("string");
    let err = validate_metadata(&metadata).unwrap_err();
    assert!(err.contains("must be a JSON object"), "got: {err}");
}

#[test]
fn reject_too_many_keys() {
    let mut map = serde_json::Map::new();
    for i in 0..17 {
        map.insert(format!("key{i}"), Value::String("val".to_owned()));
    }
    let err = validate_metadata(&Value::Object(map)).unwrap_err();
    assert!(err.contains("at most 16 keys"), "got: {err}");
}

#[test]
fn reject_long_key() {
    let long_key = "k".repeat(65);
    let metadata = serde_json::json!({long_key: "value"});
    let err = validate_metadata(&metadata).unwrap_err();
    assert!(err.contains("exceeds 64 bytes"), "got: {err}");
}

#[test]
fn reject_long_value() {
    let long_value = "v".repeat(513);
    let metadata = serde_json::json!({"key": long_value});
    let err = validate_metadata(&metadata).unwrap_err();
    assert!(err.contains("exceeds 512 bytes"), "got: {err}");
}

#[test]
fn reject_non_string_value() {
    let metadata = serde_json::json!({"key": 42});
    let err = validate_metadata(&metadata).unwrap_err();
    assert!(err.contains("must be a string"), "got: {err}");
}

// -----------------------------------------------------------------------------
// Filter Factory Tests
// -----------------------------------------------------------------------------

#[test]
fn from_config_creates_filter() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: conversations
        items_table: conversation_items
        "#,
    )
    .unwrap();
    let filter = OpenaiConversationsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_conversations");
}

// -----------------------------------------------------------------------------
// Handler Tests
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_conversation() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": {"env": "test"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(resp["object"], "conversation");
    let conv_id = resp["id"].as_str().unwrap();
    assert!(conv_id.starts_with("conv_"));

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(resp["id"], conv_id);
    assert_eq!(resp["metadata"]["env"], "test");
}

#[tokio::test]
async fn get_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::GET, "/v1/conversations/conv_nonexistent");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 404);
}

#[tokio::test]
async fn update_conversation() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"v": "1"})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": {"v": "2"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(resp["metadata"]["v"], "2");

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get after update");
    };
    let resp = rejection_body(&rejection);
    assert_eq!(resp["metadata"]["v"], "2", "updated metadata should be persisted");
}

#[tokio::test]
async fn update_conversation_without_metadata_preserves_existing_metadata() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"v": "1"})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{}"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["metadata"]["v"], "1",
        "missing metadata should preserve existing value"
    );

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get after update");
    };
    let resp = rejection_body(&rejection);
    assert_eq!(resp["metadata"]["v"], "1", "preserved metadata should be persisted");
}

#[tokio::test]
async fn delete_conversation() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::DELETE, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert!(resp["deleted"].as_bool().unwrap());

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject");
    };
    assert_eq!(rejection.status, 404);
}

#[tokio::test]
async fn delete_conversation_preserves_item_rows() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "metadata": {},
        "items": [
            {"id": "item_keep", "type": "message", "role": "user", "content": "keep me"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create conversation");
    };
    assert_eq!(rejection.status, 200, "create should return 200");
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(Method::DELETE, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from delete conversation");
    };
    assert_eq!(rejection.status, 200, "delete conversation should return 200");

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items/item_keep"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get retained item");
    };
    assert_eq!(rejection.status, 200, "conversation delete should not delete item row");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["id"], "item_keep");
    assert_eq!(resp["content"][0]["text"], "keep me");
}

#[tokio::test]
async fn unmatched_path_continues() {
    let filter = build_test_filter();

    let req = make_request(Method::GET, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test]
async fn post_routes_use_stream_buffer_body_mode() {
    let filter = build_test_filter();
    assert!(
        matches!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer { max_bytes: Some(_) }
        ),
        "conversation POST routes require buffered bodies for local handling"
    );

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    ctx.request_body_mode = filter.request_body_mode();
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(
        matches!(ctx.request_body_mode, BodyMode::StreamBuffer { max_bytes: Some(_) }),
        "matched POST should keep buffering enabled for request-body handling"
    );
}

#[tokio::test]
async fn unmatched_post_path_continues() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(
        matches!(
            filter.request_body_mode(),
            BodyMode::StreamBuffer { max_bytes: Some(_) }
        ),
        "body mode declaration is static; unmatched path handling remains a local Continue"
    );
}

#[tokio::test]
async fn early_body_pre_read_defers_store_write_until_request_filters_run() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(7);

    let body_json = serde_json::json!({"metadata": {"phase": "deferred"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "early body hook should not write the store before request filters run"
    );

    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected deferred body to be handled during on_request, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(resp["metadata"]["phase"], "deferred");
}

#[tokio::test]
async fn create_conversation_with_invalid_metadata() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": "not-an-object"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for invalid metadata, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "invalid metadata should return 400");
}

#[tokio::test]
async fn create_conversation_with_invalid_json_returns_400() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{not-json"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for invalid JSON, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "invalid JSON should return 400");
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["error"]["type"], "invalid_request_error",
        "invalid JSON should be a client error"
    );
}

#[tokio::test]
async fn create_conversation_with_non_object_json_returns_400() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"[]"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for non-object JSON, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-object JSON should return 400");
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["error"]["type"], "invalid_request_error",
        "non-object JSON should be a client error"
    );
}

#[tokio::test]
async fn update_conversation_with_non_object_json_preserves_metadata() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"v": "1"})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"[]"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for non-object JSON, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-object JSON should return 400");

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get after invalid update");
    };
    let resp = rejection_body(&rejection);
    assert_eq!(resp["metadata"]["v"], "1", "invalid update should not reset metadata");
}

#[tokio::test]
async fn initial_items_can_be_listed_and_retrieved() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "metadata": {},
        "items": [
            {"id": "item_explicit", "type": "message", "role": "user", "content": "hello"},
            {"type": "message", "role": "assistant", "content": "hi"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create conversation");
    };
    assert_eq!(rejection.status, 200, "create should return 200");
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items?order=asc"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list items");
    };
    assert_eq!(rejection.status, 200, "list items should return 200");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"][0]["id"], "item_explicit");
    assert_eq!(resp["data"][0]["status"], "completed");
    assert_eq!(resp["data"][0]["content"][0]["type"], "input_text");
    assert_eq!(resp["data"][0]["content"][0]["text"], "hello");
    let generated_id = resp["data"][1]["id"].as_str().unwrap();
    assert!(generated_id.starts_with("item_"), "missing item ID should be generated");
    assert_eq!(resp["data"][1]["status"], "completed");
    assert_eq!(resp["data"][1]["content"][0]["type"], "output_text");
    assert_eq!(resp["data"][1]["content"][0]["text"], "hi");
    assert_eq!(resp["data"][1]["content"][0]["annotations"], serde_json::json!([]));

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items/item_explicit"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get item");
    };
    assert_eq!(rejection.status, 200, "get item should return 200");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["status"], "completed");
    assert_eq!(resp["content"][0]["type"], "input_text");
    assert_eq!(resp["content"][0]["text"], "hello");
}

#[tokio::test]
async fn empty_item_list_returns_string_pagination_ids() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list items");
    };
    assert_eq!(rejection.status, 200, "list empty items should return 200");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"], serde_json::json!([]));
    assert_eq!(resp["first_id"], "");
    assert_eq!(resp["last_id"], "");
    assert_eq!(resp["has_more"], false);
}

#[tokio::test]
async fn create_conversation_rejects_duplicate_initial_item_ids() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "metadata": {},
        "items": [
            {"id": "item_dup", "type": "message", "role": "user", "content": "first"},
            {"id": "item_dup", "type": "message", "role": "assistant", "content": "second"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for duplicate item id");
    };
    assert_eq!(rejection.status, 400, "duplicate initial item IDs should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"].as_str().unwrap().contains("duplicate item id"),
        "duplicate error should mention item id"
    );
}

#[tokio::test]
async fn create_and_delete_item_endpoints_are_local() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_new", "type": "message", "role": "user", "content": "new"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create items");
    };
    assert_eq!(rejection.status, 200, "create items should return 200");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"][0]["id"], "item_new");
    assert_eq!(resp["data"][0]["status"], "completed");
    assert_eq!(resp["data"][0]["content"][0]["type"], "input_text");
    assert_eq!(resp["data"][0]["content"][0]["text"], "new");

    let req = make_request(Method::DELETE, &format!("/v1/conversations/{conv_id}/items/item_new"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from delete item");
    };
    assert_eq!(rejection.status, 200, "delete item should return 200");

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items/item_new"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get deleted item");
    };
    assert_eq!(rejection.status, 404, "deleted item should return 404");
}

#[tokio::test]
async fn item_subresource_routes_do_not_fall_through_upstream() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST item route should continue only until request-body handling"
    );
    assert!(
        matches!(ctx.request_body_mode, BodyMode::StreamBuffer { max_bytes: Some(_) }),
        "POST item route should keep body buffering so it cannot reach upstream"
    );

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_local", "type": "message", "role": "user", "content": "local"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected local Reject from POST item body");
    };
    assert_eq!(rejection.status, 200, "POST item route should be handled locally");

    for (method, path) in [
        (Method::GET, format!("/v1/conversations/{conv_id}/items")),
        (Method::GET, format!("/v1/conversations/{conv_id}/items/item_local")),
        (Method::DELETE, format!("/v1/conversations/{conv_id}/items/item_local")),
    ] {
        let req = make_request(method.clone(), &path);
        let mut ctx = make_filter_context(&req);
        let action = filter.on_request(&mut ctx).await.unwrap();
        let FilterAction::Reject(rejection) = action else {
            panic!("{method} {path} should be handled locally, got {action:?}");
        };
        assert!(
            matches!(rejection.status, 200 | 404),
            "{method} {path} should return a local item response, got {}",
            rejection.status
        );
    }
}

#[tokio::test]
async fn encoded_item_id_path_segments_are_decoded() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item with space", "type": "message", "role": "user", "content": "encoded"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create items");
    };
    assert_eq!(rejection.status, 200, "create items should return 200");

    let req = make_request(
        Method::GET,
        &format!("/v1/conversations/{conv_id}/items/item%20with%20space"),
    );
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get encoded item");
    };
    assert_eq!(rejection.status, 200, "encoded item ID should be retrievable");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["id"], "item with space");
    assert_eq!(resp["content"][0]["text"], "encoded");

    let req = make_request(
        Method::DELETE,
        &format!("/v1/conversations/{conv_id}/items/item%20with%20space"),
    );
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from delete encoded item");
    };
    assert_eq!(rejection.status, 200, "encoded item ID should be deletable");
}

#[tokio::test]
async fn item_list_after_cursor_decodes_query_plus_as_space() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item with space", "type": "message", "role": "user", "content": "first"},
            {"id": "item_next", "type": "message", "role": "assistant", "content": "second"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create conversation");
    };
    assert_eq!(rejection.status, 200, "create should return 200");
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(
        Method::GET,
        &format!("/v1/conversations/{conv_id}/items?order=asc&after=item+with+space"),
    );
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list after cursor");
    };
    assert_eq!(rejection.status, 200, "list items should return 200");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"].as_array().unwrap().len(), 1);
    assert_eq!(resp["data"][0]["id"], "item_next");
    assert_eq!(resp["data"][0]["content"][0]["text"], "second");
}

#[tokio::test]
async fn create_items_rejects_duplicate_ids_in_request() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_dup", "type": "message", "role": "user", "content": "first"},
            {"id": "item_dup", "type": "message", "role": "assistant", "content": "second"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for duplicate item id");
    };
    assert_eq!(rejection.status, 400, "duplicate request item IDs should return 400");
}

#[tokio::test]
async fn create_items_rejects_existing_id_without_overwrite() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    let body_json = serde_json::json!({
        "items": [
            {"id": "item_existing", "type": "message", "role": "user", "content": "original"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from initial item create");
    };
    assert_eq!(rejection.status, 200, "initial item create should succeed");

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());
    let body_json = serde_json::json!({
        "items": [
            {"id": "item_existing", "type": "message", "role": "assistant", "content": "overwrite"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for existing item id");
    };
    assert_eq!(rejection.status, 400, "existing item ID should return 400");

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items/item_existing"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from get existing item");
    };
    assert_eq!(rejection.status, 200, "original item should still exist");
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["content"][0]["text"], "original",
        "duplicate create must not overwrite item data"
    );
}

// -----------------------------------------------------------------------------
// Handler Tests — Delete Non-existent
// -----------------------------------------------------------------------------

#[tokio::test]
async fn delete_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::DELETE, "/v1/conversations/conv_nonexistent");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 404);
}

#[tokio::test]
async fn update_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations/conv_nonexistent");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": {"v": "1"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 404);
}

// -----------------------------------------------------------------------------
// Handler Tests — Item Create Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_items_missing_items_field_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{}"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for missing items, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "missing items should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"].as_str().unwrap().contains("required"),
        "should mention items is required"
    );
}

#[tokio::test]
async fn create_items_non_array_items_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"items": "not-an-array"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for non-array items, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-array items should return 400");
}

#[tokio::test]
async fn create_items_null_items_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(br#"{"items":null}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for null items, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "null items should return 400");
}

#[tokio::test]
async fn create_items_too_many_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let items: Vec<Value> = (0..21)
        .map(|i| serde_json::json!({"id": format!("item_{i}"), "type": "message", "role": "user", "content": "hi"}))
        .collect();
    let body_json = serde_json::json!({"items": items});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for too many items, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "too many items should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"].as_str().unwrap().contains("at most"),
        "should mention items limit"
    );
}

#[tokio::test]
async fn create_items_for_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations/conv_nonexistent/items");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_1", "type": "message", "role": "user", "content": "hi"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 404, "non-existent conversation should return 404");
}

#[tokio::test]
async fn create_items_with_invalid_json_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{not-json"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for invalid JSON, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "invalid JSON should return 400");
}

#[tokio::test]
async fn create_items_with_non_object_json_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"[]"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for non-object JSON, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-object JSON should return 400");
}

// -----------------------------------------------------------------------------
// Handler Tests — Item Normalization Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_items_with_non_object_item_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"items": ["not-an-object"]});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for non-object item, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-object item should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("must be a JSON object"),
        "should mention object requirement"
    );
}

#[tokio::test]
async fn create_items_with_empty_item_id_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"id": "", "type": "message", "role": "user", "content": "hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for empty item id, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "empty item id should return 400");
}

#[tokio::test]
async fn create_items_with_numeric_item_id_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"id": 42, "type": "message", "role": "user", "content": "hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for numeric item id, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "numeric item id should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"].as_str().unwrap().contains("must be a string"),
        "should mention string requirement"
    );
}

#[tokio::test]
async fn create_items_with_null_item_id_generates_id() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"id": null, "type": "message", "role": "user", "content": "hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200, "null item id should auto-generate");
    let resp = rejection_body(&rejection);
    let generated_id = resp["data"][0]["id"].as_str().unwrap();
    assert!(
        generated_id.starts_with("item_"),
        "generated id should have item_ prefix"
    );
}

// -----------------------------------------------------------------------------
// Handler Tests — Message Role/Content Validation
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_items_with_empty_role_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"type": "message", "role": "", "content": "hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "empty role should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"].as_str().unwrap().contains("role"),
        "empty role error should mention role: {resp}"
    );
}

#[tokio::test]
async fn create_items_with_non_string_role_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"type": "message", "role": 42, "content": "hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-string role should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("role must be a string")
    );
}

#[tokio::test]
async fn create_items_with_missing_role_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"type": "message", "content": "hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "missing role should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"].as_str().unwrap().contains("role is required"),
        "missing role error should mention required role: {resp}"
    );
}

#[tokio::test]
async fn create_items_with_missing_content_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"type": "message", "role": "user"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "missing content should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("content is required")
    );
}

#[tokio::test]
async fn create_items_with_non_string_non_array_content_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"type": "message", "role": "user", "content": 42}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "numeric content should return 400");
    let resp = rejection_body(&rejection);
    assert!(
        resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("must be a string or array")
    );
}

#[tokio::test]
async fn create_items_with_array_content_passthrough() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let content = serde_json::json!([{"type": "input_text", "text": "array content"}]);
    let body_json = serde_json::json!({
        "items": [{"type": "message", "role": "user", "content": content}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200, "array content should be accepted");
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["data"][0]["content"][0]["text"], "array content",
        "array content should pass through unchanged"
    );
}

#[tokio::test]
async fn non_message_item_type_skips_normalization() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"type": "function_call", "name": "test", "arguments": "{}"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200, "non-message type should be accepted");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"][0]["type"], "function_call");
}

// -----------------------------------------------------------------------------
// Handler Tests — Item Delete Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn delete_item_from_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::DELETE, "/v1/conversations/conv_nonexistent/items/item_1");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(
        rejection.status, 404,
        "delete item from non-existent conversation should return 404"
    );
}

#[tokio::test]
async fn delete_nonexistent_item_returns_404() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(
        Method::DELETE,
        &format!("/v1/conversations/{conv_id}/items/item_nonexistent"),
    );
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 404, "delete non-existent item should return 404");
}

#[tokio::test]
async fn get_item_from_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::GET, "/v1/conversations/conv_nonexistent/items/item_1");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(
        rejection.status, 404,
        "get item from non-existent conversation should return 404"
    );
}

// -----------------------------------------------------------------------------
// Handler Tests — List Items Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn list_items_for_nonexistent_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::GET, "/v1/conversations/conv_nonexistent/items");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(
        rejection.status, 404,
        "list items for non-existent conversation should return 404"
    );
}

#[tokio::test]
async fn list_items_with_limit_parameter() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_a", "type": "message", "role": "user", "content": "first"},
            {"id": "item_b", "type": "message", "role": "assistant", "content": "second"},
            {"id": "item_c", "type": "message", "role": "user", "content": "third"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create");
    };
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(
        Method::GET,
        &format!("/v1/conversations/{conv_id}/items?limit=2&order=asc"),
    );
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list items");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"].as_array().unwrap().len(), 2, "should respect limit");
    assert_eq!(resp["has_more"], true, "should indicate more items");
    assert_eq!(resp["data"][0]["id"], "item_a");
    assert_eq!(resp["data"][1]["id"], "item_b");
}

#[tokio::test]
async fn list_items_desc_order() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_a", "type": "message", "role": "user", "content": "first"},
            {"id": "item_b", "type": "message", "role": "assistant", "content": "second"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create");
    };
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items?order=desc"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list items");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(resp["data"][0]["id"], "item_b", "desc order should list newest first");
    assert_eq!(resp["data"][1]["id"], "item_a");
}

// -----------------------------------------------------------------------------
// Handler Tests — Conversation Create with Initial Items Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn create_conversation_with_non_array_items_returns_400() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"items": "not-an-array"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "non-array items should return 400");
}

#[tokio::test]
async fn create_conversation_with_null_items_returns_400() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(br#"{"items":null}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject for null items, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "null items should return 400");
}

#[tokio::test]
async fn create_conversation_with_too_many_initial_items_returns_400() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let items: Vec<Value> = (0..21)
        .map(|i| serde_json::json!({"id": format!("item_{i}"), "type": "message", "role": "user", "content": "hi"}))
        .collect();
    let body_json = serde_json::json!({"items": items});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "too many initial items should return 400");
}

#[tokio::test]
async fn create_conversation_with_null_metadata_defaults_to_empty() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": null});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["metadata"],
        serde_json::json!({}),
        "null metadata should default to empty object"
    );
}

#[tokio::test]
async fn create_conversation_without_body_metadata_defaults_to_empty() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{}"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["metadata"],
        serde_json::json!({}),
        "missing metadata should default to empty object"
    );
}

// -----------------------------------------------------------------------------
// Handler Tests — Update Conversation Edge Cases
// -----------------------------------------------------------------------------

#[tokio::test]
async fn update_conversation_with_invalid_metadata_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"v": "1"})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": "not-an-object"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "invalid metadata should return 400");
}

#[tokio::test]
async fn update_conversation_with_null_metadata_clears_metadata() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"v": "1"})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": null});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["metadata"],
        serde_json::json!({}),
        "null metadata should clear to empty object"
    );
}

#[tokio::test]
async fn update_conversation_with_invalid_json_returns_400() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"v": "1"})).await;

    let req = make_request(Method::POST, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{bad-json"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 400, "invalid JSON in update should return 400");
}

// -----------------------------------------------------------------------------
// Handler Tests — Tenant Isolation
// -----------------------------------------------------------------------------

#[tokio::test]
async fn cross_tenant_get_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.tenant_id", "tenant-a");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": {"owner": "a"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create");
    };
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.tenant_id", "tenant-b");
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from cross-tenant GET");
    };
    assert_eq!(
        rejection.status, 404,
        "cross-tenant GET should return 404, not leak data"
    );
}

#[tokio::test]
async fn cross_tenant_delete_conversation_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.tenant_id", "tenant-a");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": {"owner": "a"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create");
    };
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(Method::DELETE, &format!("/v1/conversations/{conv_id}"));
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.tenant_id", "tenant-b");
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from cross-tenant DELETE");
    };
    assert_eq!(
        rejection.status, 404,
        "cross-tenant DELETE should return 404, not delete another tenant's data"
    );
}

#[tokio::test]
async fn cross_tenant_delete_item_returns_404() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.tenant_id", "tenant-a");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [{"id": "item_secret", "type": "message", "role": "user", "content": "private"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create");
    };
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(
        Method::DELETE,
        &format!("/v1/conversations/{conv_id}/items/item_secret"),
    );
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.tenant_id", "tenant-b");
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from cross-tenant item DELETE");
    };
    assert_eq!(rejection.status, 404, "cross-tenant item DELETE should return 404");
}

// -----------------------------------------------------------------------------
// Handler Tests — Delete Item Syncs Conversation Messages
// -----------------------------------------------------------------------------

#[tokio::test]
async fn delete_item_returns_updated_conversation() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({
        "items": [
            {"id": "item_stay", "type": "message", "role": "user", "content": "keep"},
            {"id": "item_gone", "type": "message", "role": "assistant", "content": "remove"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create");
    };
    let resp = rejection_body(&rejection);
    let conv_id = resp["id"].as_str().unwrap();

    let req = make_request(Method::DELETE, &format!("/v1/conversations/{conv_id}/items/item_gone"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from delete item");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    assert_eq!(
        resp["object"], "conversation",
        "delete item should return updated conversation"
    );
    assert_eq!(resp["id"], conv_id);

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list items");
    };
    assert_eq!(rejection.status, 200);
    let items_resp = rejection_body(&rejection);
    let items = items_resp["data"].as_array().unwrap();
    assert_eq!(items.len(), 1, "only the kept item should remain");
    assert_eq!(items[0]["id"], "item_stay");
}

// -----------------------------------------------------------------------------
// Filter Tests — Body Modes and Access
// -----------------------------------------------------------------------------

#[test]
fn filter_request_body_access_is_read_only() {
    let filter = build_test_filter();
    assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
}

// -----------------------------------------------------------------------------
// Filter Tests — Trailing Slash Normalization
// -----------------------------------------------------------------------------

#[tokio::test]
async fn trailing_slash_on_conversation_path_is_normalized() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({"k": "v"})).await;

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200, "trailing slash should be normalized");
    let resp = rejection_body(&rejection);
    assert_eq!(resp["id"], conv_id);
}

#[tokio::test]
async fn trailing_slash_on_items_path_is_normalized() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items/"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(
        rejection.status, 200,
        "trailing slash on items path should be normalized"
    );
}

// -----------------------------------------------------------------------------
// Filter Tests — Non-POST Body Passthrough
// -----------------------------------------------------------------------------

#[tokio::test]
async fn body_hook_with_end_of_stream_false_continues() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"partial"));
    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-final body chunk should continue"
    );
}

#[tokio::test]
async fn body_hook_for_get_request_continues() {
    let filter = build_test_filter();

    let req = make_request(Method::GET, "/v1/conversations/conv_1");
    let mut ctx = make_filter_context(&req);

    let mut body = Some(Bytes::from_static(b"ignored"));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "body hook for GET should continue"
    );
}

// -----------------------------------------------------------------------------
// Filter Tests — Unmatched Methods
// -----------------------------------------------------------------------------

#[tokio::test]
async fn put_on_conversation_path_continues() {
    let filter = build_test_filter();

    let req = make_request(Method::PUT, "/v1/conversations/conv_1");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "PUT should not be handled");
}

#[tokio::test]
async fn patch_on_conversation_path_continues() {
    let filter = build_test_filter();

    let req = make_request(Method::PATCH, "/v1/conversations/conv_1");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "PATCH should not be handled");
}

// -----------------------------------------------------------------------------
// Append-Back: on_response
// -----------------------------------------------------------------------------

fn set_append_back_metadata(ctx: &mut praxis_filter::HttpFilterContext<'_>) {
    ctx.set_metadata("openai_responses_format.has_conversation", "true");
    ctx.set_metadata("responses.conversation_id", "conv_test_123");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_not_armed_without_conversation_metadata() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_not_armed_when_streaming() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);
    ctx.set_metadata("openai_responses_format.stream", "true");

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_not_armed_when_background() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);
    ctx.set_metadata("openai_responses_format.background", "true");

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_not_armed_for_non_2xx() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.status = http::StatusCode::INTERNAL_SERVER_ERROR;
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_not_armed_for_non_json_content_type() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "text/plain".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_armed_for_json_200() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

// -----------------------------------------------------------------------------
// Append-Back: on_response_body
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_releases_when_not_armed() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);

    drop(filter.on_response(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{}"));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Release));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_continues_when_not_end_of_stream() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"partial"));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_non_completed_status() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let response_json = serde_json::json!({
        "status": "failed",
        "output": [{"type": "message", "role": "assistant", "content": "oops"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_invalid_json() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from_static(b"{not-json"));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_empty_items() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let response_json = serde_json::json!({
        "status": "completed",
        "output": []
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_empty_body() {
    let filter = build_test_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    set_append_back_metadata(&mut ctx);

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_appends_completed_response() {
    let filter = build_test_filter();
    let conv_id = create_test_conversation(filter.as_ref(), serde_json::json!({})).await;

    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.set_metadata("openai_responses_format.has_conversation", "true");
    ctx.set_metadata("responses.conversation_id", &conv_id);

    let input_items = vec![serde_json::json!({
        "type": "message",
        "role": "user",
        "content": "hello from append"
    })];
    ctx.extensions.insert(ResponsesState {
        input: input_items,
        ..ResponsesState::default()
    });

    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut resp = make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let response_json = serde_json::json!({
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": "hi from model"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let req = make_request(Method::GET, &format!("/v1/conversations/{conv_id}/items?order=asc"));
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from list items after append-back");
    };
    assert_eq!(rejection.status, 200);
    let resp = rejection_body(&rejection);
    let items = resp["data"].as_array().unwrap();
    assert_eq!(items.len(), 2, "append-back should persist both input and output items");
}

#[test]
fn conformance_conversations_routes_match_runtime_registry() {
    let spec = generated_openapi_spec();
    assert_eq!(
        generated_operation_keys(&spec),
        route_operation_keys(),
        "generated OpenAPI paths must exactly match Conversations operation_specs()"
    );

    for operation in operation_specs() {
        let path = runtime_path(operation, Some("conv_sync"), Some("item_sync"));
        let matched = routes::match_route(operation.method, &path)
            .unwrap_or_else(|| panic!("runtime route table did not match {} {path}", operation.method));
        assert_eq!(
            matched.spec.operation, operation.operation,
            "runtime route table matched the wrong operation for {} {path}",
            operation.method,
        );
        assert_eq!(
            OperationKey::new(matched.spec.method, matched.spec.spec_path),
            OperationKey::new(operation.method, operation.spec_path),
            "runtime route metadata drifted from operation_specs() for {} {path}",
            operation.method,
        );
    }
    println!("PRAXIS_CONFORMANCE_OK conversations route_dispatch");
}

#[tokio::test]
async fn conformance_conversations_success_payloads_match_generated_response_schemas() {
    let filter = build_test_filter();
    let spec = generated_openapi_spec();
    let schemas = generated_response_schemas(&spec);
    let payloads = successful_conversation_payloads(filter.as_ref()).await;

    let schema_operations: Vec<_> = schemas.keys().collect();
    let payload_operations: Vec<_> = payloads.keys().collect();
    assert_eq!(
        payload_operations, schema_operations,
        "runtime success fixtures should cover every generated Conversations response schema"
    );

    for (operation, schema) in schemas {
        let payload = payloads
            .get(&operation)
            .unwrap_or_else(|| panic!("missing runtime success payload for {}", operation.label()));
        assert_matches_schema(&spec, &format!("{} response", operation.label()), &schema, payload);
    }
    println!("PRAXIS_CONFORMANCE_OK conversations success_response_contract");
}

#[test]
fn conformance_conversations_generated_schema_check_rejects_wrong_discriminator() {
    let spec = generated_openapi_spec();
    let schema = spec
        .pointer("/components/schemas/ConversationResource")
        .unwrap_or_else(|| panic!("generated OpenAPI missing ConversationResource"));
    let invalid = serde_json::json!({
        "id": "conv_invalid",
        "object": "wrong",
        "created_at": 1,
        "metadata": {},
    });

    assert!(
        !schema_matches_value(&spec, schema, &invalid),
        "schema check must reject a response with the wrong object discriminator"
    );
    println!("PRAXIS_CONFORMANCE_OK conversations schema_check_sensitivity");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn build_test_filter() -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        backend: sqlite
        database_url: "sqlite::memory:"
        conversations_table: test_conversations
        items_table: test_items
        "#,
    )
    .unwrap();
    OpenaiConversationsFilter::from_config(&yaml).unwrap()
}

async fn create_test_conversation(filter: &dyn HttpFilter, metadata: Value) -> String {
    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": metadata});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject from create conversation");
    };
    let resp = rejection_body(&rejection);
    resp["id"].as_str().unwrap().to_owned()
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OperationKey {
    method: String,
    path: String,
}

impl OperationKey {
    fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
        }
    }

    fn label(&self) -> String {
        format!("{} {}", self.method, self.path)
    }
}

async fn successful_conversation_payloads(filter: &dyn HttpFilter) -> BTreeMap<OperationKey, Value> {
    let mut payloads = BTreeMap::new();
    let create_spec = operation_spec(ConversationOperation::CreateConversation);

    let create = successful_post_json(
        filter,
        &runtime_path(create_spec, None, None),
        serde_json::json!({"metadata": {"project": "schema-test"}}),
    )
    .await;
    let conv_id = create["id"].as_str().unwrap().to_owned();
    insert_payload(&mut payloads, create_spec, create);

    let get_spec = operation_spec(ConversationOperation::GetConversation);
    let get = successful_request_json(filter, Method::GET, &runtime_path(get_spec, Some(&conv_id), None)).await;
    insert_payload(&mut payloads, get_spec, get);

    let update_spec = operation_spec(ConversationOperation::UpdateConversation);
    let update = successful_post_json(
        filter,
        &runtime_path(update_spec, Some(&conv_id), None),
        serde_json::json!({"metadata": {"project": "updated"}}),
    )
    .await;
    insert_payload(&mut payloads, update_spec, update);

    let create_items_spec = operation_spec(ConversationOperation::CreateConversationItems);
    let create_items = successful_post_json(
        filter,
        &runtime_path(create_items_spec, Some(&conv_id), None),
        serde_json::json!({
            "items": [
                {"id": "item_schema", "type": "message", "role": "user", "content": "hello"}
            ]
        }),
    )
    .await;
    insert_payload(&mut payloads, create_items_spec, create_items);

    let list_spec = operation_spec(ConversationOperation::ListConversationItems);
    let list = successful_request_json(filter, Method::GET, &runtime_path(list_spec, Some(&conv_id), None)).await;
    insert_payload(&mut payloads, list_spec, list);

    let get_item_spec = operation_spec(ConversationOperation::GetConversationItem);
    let item = successful_request_json(
        filter,
        Method::GET,
        &runtime_path(get_item_spec, Some(&conv_id), Some("item_schema")),
    )
    .await;
    insert_payload(&mut payloads, get_item_spec, item);

    let delete_item_spec = operation_spec(ConversationOperation::DeleteConversationItem);
    let delete_item = successful_request_json(
        filter,
        Method::DELETE,
        &runtime_path(delete_item_spec, Some(&conv_id), Some("item_schema")),
    )
    .await;
    insert_payload(&mut payloads, delete_item_spec, delete_item);

    let delete_spec = operation_spec(ConversationOperation::DeleteConversation);
    let delete =
        successful_request_json(filter, Method::DELETE, &runtime_path(delete_spec, Some(&conv_id), None)).await;
    insert_payload(&mut payloads, delete_spec, delete);

    payloads
}

fn operation_spec(operation: ConversationOperation) -> &'static ConversationOperationSpec {
    operation_specs()
        .iter()
        .find(|spec| spec.operation == operation)
        .unwrap_or_else(|| panic!("missing operation spec for {operation:?}"))
}

fn runtime_path(spec: &ConversationOperationSpec, conversation_id: Option<&str>, item_id: Option<&str>) -> String {
    spec.runtime_path
        .replace("{conversation_id}", conversation_id.unwrap_or_default())
        .replace("{item_id}", item_id.unwrap_or_default())
}

fn insert_payload(payloads: &mut BTreeMap<OperationKey, Value>, spec: &ConversationOperationSpec, payload: Value) {
    let previous = payloads.insert(OperationKey::new(spec.method, spec.spec_path), payload);
    assert!(
        previous.is_none(),
        "duplicate runtime success fixture for {} {}",
        spec.method,
        spec.spec_path
    );
}

async fn successful_post_json(filter: &dyn HttpFilter, path: &str, body_json: Value) -> Value {
    let req = make_request(Method::POST, path);
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    rejection_body(&rejection)
}

async fn successful_request_json(filter: &dyn HttpFilter, method: Method, path: &str) -> Value {
    let req = make_request(method, path);
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    assert_eq!(rejection.status, 200);
    rejection_body(&rejection)
}

fn generated_openapi_spec() -> Value {
    serde_json::from_str(&super::implementation_openapi_json().unwrap()).unwrap()
}

fn route_operation_keys() -> Vec<OperationKey> {
    let mut keys = operation_specs()
        .iter()
        .map(|spec| OperationKey::new(spec.method, spec.spec_path))
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

fn generated_operation_keys(spec: &Value) -> Vec<OperationKey> {
    let mut keys = Vec::new();
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("generated OpenAPI spec missing paths object"));

    for (path, path_item) in paths {
        let path_item = path_item
            .as_object()
            .unwrap_or_else(|| panic!("generated path item {path} should be an object"));
        for (method, _operation) in path_item {
            if matches!(method.as_str(), "delete" | "get" | "post") {
                keys.push(OperationKey::new(method.to_uppercase(), path));
            }
        }
    }

    keys.sort();
    keys
}

fn generated_response_schemas(spec: &Value) -> BTreeMap<OperationKey, Value> {
    let mut schemas = BTreeMap::new();
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("generated OpenAPI spec missing paths object"));

    for (path, path_item) in paths {
        let path_item = path_item
            .as_object()
            .unwrap_or_else(|| panic!("generated path item {path} should be an object"));
        for (method, operation) in path_item {
            if !matches!(method.as_str(), "delete" | "get" | "post") {
                continue;
            }

            let schema = operation
                .pointer("/responses/200/content/application~1json/schema")
                .unwrap_or_else(|| panic!("generated operation {method} {path} missing 200 JSON response schema"));
            let previous = schemas.insert(OperationKey::new(method.to_uppercase(), path), schema.clone());
            assert!(previous.is_none(), "duplicate generated operation {method} {path}");
        }
    }

    let schema_operations = schemas.keys().cloned().collect::<Vec<_>>();
    assert_eq!(
        schema_operations,
        route_operation_keys(),
        "generated OpenAPI operations should match Conversations operation_specs()"
    );

    schemas
}

fn assert_matches_schema(spec: &Value, path: &str, schema: &Value, value: &Value) {
    assert!(
        schema_matches_value(spec, schema, value),
        "{path} does not match generated schema: {value}"
    );
    let schema = resolve_schema_ref(spec, schema);

    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
        assert!(
            enum_values.contains(value),
            "{path} should be one of {enum_values:?}, got {value}"
        );
    }

    if let Some(schema_type) = schema.get("type").and_then(Value::as_str) {
        assert_schema_type(path, schema_type, value);
    } else if schema.get("properties").is_some() || schema.get("required").is_some() {
        assert_schema_type(path, "object", value);
    }

    if let Some(items_schema) = schema.get("items") {
        for (idx, item) in value
            .as_array()
            .unwrap_or_else(|| panic!("{path} should be an array"))
            .iter()
            .enumerate()
        {
            assert_matches_schema(spec, &format!("{path}[{idx}]"), items_schema, item);
        }
    }

    let Some(obj) = value.as_object() else {
        return;
    };

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for required_name in required {
            let required_name = required_name.as_str().unwrap();
            assert!(
                obj.contains_key(required_name),
                "{path} missing required property {required_name}"
            );
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (property_name, property_schema) in properties {
            if let Some(property_value) = obj.get(property_name) {
                assert_matches_schema(
                    spec,
                    &format!("{path}.{property_name}"),
                    property_schema,
                    property_value,
                );
            }
        }
    }

    if let Some(additional_schema) = schema.get("additionalProperties").filter(|schema| schema.is_object()) {
        let properties = schema.get("properties").and_then(Value::as_object);
        for (property_name, property_value) in obj {
            if properties.is_none_or(|properties| !properties.contains_key(property_name)) {
                assert_matches_schema(
                    spec,
                    &format!("{path}.{property_name}"),
                    additional_schema,
                    property_value,
                );
            }
        }
    }
}

fn schema_matches_value(spec: &Value, schema: &Value, value: &Value) -> bool {
    let schema = resolve_schema_ref(spec, schema);

    if value.is_null() && schema.get("nullable").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array)
        && variants
            .iter()
            .filter(|variant| schema_matches_value(spec, variant, value))
            .count()
            != 1
    {
        return false;
    }
    if let Some(variants) = schema.get("anyOf").and_then(Value::as_array)
        && !variants
            .iter()
            .any(|variant| schema_matches_value(spec, variant, value))
    {
        return false;
    }
    if let Some(variants) = schema.get("allOf").and_then(Value::as_array)
        && !variants
            .iter()
            .all(|variant| schema_matches_value(spec, variant, value))
    {
        return false;
    }
    if schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| !values.contains(value))
    {
        return false;
    }
    if let Some(schema_type) = schema.get("type") {
        let matches_type = schema_type.as_str().map_or_else(
            || {
                schema_type.as_array().is_some_and(|types| {
                    types
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|kind| value_has_type(value, kind))
                })
            },
            |kind| value_has_type(value, kind),
        );
        if !matches_type {
            return false;
        }
    }

    if let Some(items_schema) = schema.get("items") {
        let Some(items) = value.as_array() else {
            return false;
        };
        if !items.iter().all(|item| schema_matches_value(spec, items_schema, item)) {
            return false;
        }
    }

    let Some(object) = value.as_object() else {
        return schema.get("properties").is_none() && schema.get("required").is_none();
    };
    if schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .any(|name| !object.contains_key(name))
        })
    {
        return false;
    }

    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties
        && properties.iter().any(|(name, property_schema)| {
            object
                .get(name)
                .is_some_and(|property| !schema_matches_value(spec, property_schema, property))
        })
    {
        return false;
    }

    match schema.get("additionalProperties") {
        Some(Value::Bool(false)) => {
            properties.is_none_or(|properties| object.keys().all(|name| properties.contains_key(name)))
        },
        Some(additional_schema @ Value::Object(_)) => object.iter().all(|(name, property)| {
            properties.is_some_and(|properties| properties.contains_key(name))
                || schema_matches_value(spec, additional_schema, property)
        }),
        _ => true,
    }
}

fn resolve_schema_ref<'a>(spec: &'a Value, schema: &'a Value) -> &'a Value {
    let Some(ref_path) = schema.get("$ref").and_then(Value::as_str) else {
        return schema;
    };
    let pointer = ref_path.strip_prefix('#').unwrap_or(ref_path);
    spec.pointer(pointer)
        .unwrap_or_else(|| panic!("missing generated schema ref {ref_path}"))
}

fn assert_schema_type(path: &str, schema_type: &str, value: &Value) {
    let matches = value_has_type(value, schema_type);
    assert!(matches, "{path} should be {schema_type}, got {value}");
}

fn value_has_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.as_f64().is_some(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => false,
    }
}

#[tokio::test]
async fn create_conversation_response_field_order_matches_openai() {
    let filter = build_test_filter();

    let req = make_request(Method::POST, "/v1/conversations");
    let mut ctx = make_filter_context(&req);
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = serde_json::json!({"metadata": {"project": "test"}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("expected Reject, got {action:?}");
    };
    let resp = rejection_body(&rejection);
    let keys: Vec<&String> = resp.as_object().unwrap().keys().collect();
    assert_eq!(keys, &["id", "object", "created_at", "metadata"]);
}
