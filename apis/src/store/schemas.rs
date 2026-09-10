// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SQL schema generation for the response store.

use super::types::StoreError;

// -----------------------------------------------------------------------------
// Table Names
// -----------------------------------------------------------------------------

/// Resolved table names for a store instance.
///
/// Table names are configured via YAML (e.g.,
/// `openai_responses`, `google_interactions`). Each provider
/// chooses its own names.
pub(crate) struct TableNames {
    /// Responses table name.
    pub responses: String,
    /// Conversation messages table name.
    pub conversations: String,
    /// Conversation items table name (optional; only used by
    /// the conversations filter).
    pub items: Option<String>,
}

// -----------------------------------------------------------------------------
// Schema Version
// -----------------------------------------------------------------------------

/// Current schema version. Bump this when the DDL changes.
pub(crate) const SCHEMA_VERSION: i64 = 1;

/// Suffix appended to the responses table name to derive the schema
/// version table name.
const SCHEMA_VERSION_SUFFIX: &str = "_schema_version";

/// Suffix appended to the responses table name to derive the
/// server-owned pending-approval table name.
const PENDING_APPROVALS_SUFFIX: &str = "_pending_approvals";

/// Derive the schema version table name from the responses table name.
pub(crate) fn schema_version_table(responses: &str) -> String {
    format!("{responses}{SCHEMA_VERSION_SUFFIX}")
}

/// Derive the pending-approvals table name from the responses table
/// name.
///
/// Like the schema version table, this is an internal derived table
/// (not configured directly), so it is always created and never
/// exposed as a YAML option.
pub(crate) fn pending_approvals_table(responses: &str) -> String {
    format!("{responses}{PENDING_APPROVALS_SUFFIX}")
}

// -----------------------------------------------------------------------------
// Schema DDL
// -----------------------------------------------------------------------------

/// Generate DDL statements for the given table names.
///
/// Each statement uses `IF NOT EXISTS` so it is safe to run on
/// every startup. The schema uses TEXT for JSON columns (standard
/// `SQLite` pattern) and BIGINT for timestamps so the same DDL is
/// compatible with `PostgreSQL` `i64` decoding.
///
/// # Errors
///
/// Returns [`StoreError::Database`] if table names contain
/// invalid characters.
#[expect(clippy::too_many_lines, reason = "linear DDL statement assembly per table")]
pub(crate) fn generate_ddl(tables: &TableNames) -> Result<Vec<String>, StoreError> {
    let (r, c) = validate_table_names(tables)?;

    let mut stmts = vec![
        responses_ddl(r),
        conversations_ddl(c),
        format!("CREATE INDEX IF NOT EXISTS idx_{c}_tenant_id ON {c}(tenant_id)"),
    ];

    if let Some(items) = &tables.items {
        let i = validate_items_table(items, r, c)?;
        append_items_ddl(&mut stmts, i);
    }

    let a = pending_approvals_table(r);
    if a.eq_ignore_ascii_case(c) {
        return Err(StoreError::Database(format!(
            "derived pending-approvals table name collides with conversation table: {a}"
        )));
    }
    if let Some(items) = &tables.items
        && a.eq_ignore_ascii_case(items)
    {
        return Err(StoreError::Database(format!(
            "derived pending-approvals table name collides with items table: {a}"
        )));
    }
    stmts.push(pending_approvals_ddl(&a));

    let v = schema_version_table(r);
    if v.eq_ignore_ascii_case(c) {
        return Err(StoreError::Database(format!(
            "derived schema version table name collides with conversation table: {v}"
        )));
    }
    if let Some(items) = &tables.items
        && v.eq_ignore_ascii_case(items)
    {
        return Err(StoreError::Database(format!(
            "derived schema version table name collides with items table: {v}"
        )));
    }
    stmts.push(format!(
        "CREATE TABLE IF NOT EXISTS {v} (version BIGINT NOT NULL PRIMARY KEY)"
    ));

    Ok(stmts)
}

/// Validate identifiers against `PostgreSQL`-specific DDL constraints.
///
/// `PostgreSQL` truncates identifiers above 63 bytes. The
/// conversation table name is also embedded in the generated tenant
/// index name, so it needs a smaller limit than table identifiers.
///
/// `PostgreSQL` also folds unquoted identifiers to lowercase, so
/// uppercase is rejected here rather than silently creating a table
/// under a name that later lookups cannot find.
///
/// Both rules are `PostgreSQL`-only; the shared identifier validation in
/// [`validate_identifier`] stays case-permissive for `SQLite`.
///
/// # Errors
///
/// Returns [`StoreError::Database`] when an identifier would exceed
/// the `PostgreSQL` limit or would be case-folded.
pub(crate) fn validate_postgres_identifiers(tables: &TableNames) -> Result<(), StoreError> {
    let (r, c) = validate_table_names(tables)?;

    validate_postgres_identifier_len("response table name", r, POSTGRES_MAX_RESPONSES_TABLE_LEN)?;
    validate_postgres_identifier_len(
        "response table name (pending-approvals suffix)",
        r,
        POSTGRES_MAX_RESPONSES_TABLE_LEN_FOR_APPROVALS,
    )?;
    validate_postgres_identifier_len("conversation table name", c, POSTGRES_MAX_CONVERSATION_TABLE_LEN)?;
    validate_postgres_identifier_case("response table name", r)?;
    validate_postgres_identifier_case("conversation table name", c)?;

    if let Some(items) = &tables.items {
        let i = validate_items_table(items, r, c)?;
        validate_postgres_identifier_len("items table name", i, POSTGRES_MAX_ITEMS_TABLE_LEN)?;
        validate_postgres_identifier_case("items table name", i)?;
    }

    Ok(())
}

/// Validate table names for a `PostgreSQL` response store.
pub(crate) fn validate_postgres_table_identifiers(
    responses_table: &str,
    conversations_table: &str,
) -> Result<(), StoreError> {
    validate_postgres_table_set_identifiers(responses_table, conversations_table, None)
}

/// Validate table identifiers for a store that may also configure
/// conversation item rows.
pub(crate) fn validate_postgres_table_set_identifiers(
    responses_table: &str,
    conversations_table: &str,
    items_table: Option<&str>,
) -> Result<(), StoreError> {
    let tables = TableNames {
        responses: responses_table.to_owned(),
        conversations: conversations_table.to_owned(),
        items: items_table.map(ToOwned::to_owned),
    };
    validate_postgres_identifiers(&tables)
}

/// DDL for the responses table.
fn responses_ddl(r: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {r} (
            tenant_id       TEXT NOT NULL,
            id              TEXT NOT NULL,
            created_at      BIGINT NOT NULL,
            model           TEXT NOT NULL,
            response_object TEXT NOT NULL,
            input           TEXT NOT NULL,
            messages        TEXT NOT NULL,
            PRIMARY KEY (tenant_id, id)
        )"
    )
}

/// DDL for the conversations table.
fn conversations_ddl(c: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {c} (
            conversation_id TEXT NOT NULL,
            tenant_id       TEXT NOT NULL,
            created_at      BIGINT NOT NULL,
            metadata        TEXT NOT NULL,
            messages        TEXT NOT NULL,
            PRIMARY KEY (conversation_id, tenant_id)
        )"
    )
}

/// Append DDL for the conversation items table and its index.
///
/// Items intentionally do not have a cascading foreign key to conversations.
/// The OpenAI Conversations API preserves items when a conversation is
/// deleted, so retention cleanup must be implemented separately from
/// `DELETE /v1/conversations/{id}`.
fn append_items_ddl(stmts: &mut Vec<String>, i: &str) {
    stmts.push(format!(
        "CREATE TABLE IF NOT EXISTS {i} (
            item_id           TEXT NOT NULL,
            tenant_id         TEXT NOT NULL,
            conversation_id   TEXT NOT NULL,
            item_data         TEXT NOT NULL,
            created_at        BIGINT NOT NULL,
            position          BIGINT NOT NULL,
            PRIMARY KEY (item_id, tenant_id, conversation_id)
        )"
    ));
    stmts.push(format!(
        "CREATE INDEX IF NOT EXISTS idx_{i}_conversation \
         ON {i}(conversation_id, tenant_id, position, item_id)"
    ));
    stmts.push(format!(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_{i}_position \
         ON {i}(tenant_id, conversation_id, position)"
    ));
}

/// DDL for the server-owned pending-approval table.
///
/// A row is written from proxy **output** the moment an
/// `mcp_approval_request` is emitted, capturing the issuing `response_id`, the
/// complete pending call (server label, tool name, arguments), and the resolved
/// target fingerprint. `consumed_at` is `NULL` while the approval is outstanding
/// and is stamped exactly once when the matching `mcp_approval_response`
/// is honored, so a replayed response cannot trigger a second tool
/// execution. The composite primary key makes insert-if-absent atomic and
/// idempotent, and consumption never resets it back to `NULL`.
///
/// `response_id` is part of the key so an approval is bound to the response that
/// issued it: a client must supply the originating `previous_response_id` to
/// load or consume it, so a fresh, unrelated request cannot claim a known
/// outstanding approval and the same model-generated call id reused across two
/// responses stays distinct.
fn pending_approvals_ddl(a: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {a} (
            tenant_id          TEXT NOT NULL,
            response_id        TEXT NOT NULL,
            approval_id        TEXT NOT NULL,
            server_label       TEXT NOT NULL,
            tool_name          TEXT NOT NULL,
            arguments          TEXT NOT NULL,
            target_fingerprint TEXT NOT NULL,
            created_at         BIGINT NOT NULL,
            consumed_at        BIGINT,
            PRIMARY KEY (tenant_id, response_id, approval_id)
        )"
    )
}

/// Validate the configured table names and return them as borrowed identifiers.
fn validate_table_names(tables: &TableNames) -> Result<(&str, &str), StoreError> {
    let r = tables.responses.as_str();
    let c = tables.conversations.as_str();

    validate_identifier(r)?;
    validate_identifier(c)?;
    if r.eq_ignore_ascii_case(c) {
        return Err(StoreError::Database(format!(
            "response and conversation table names must be distinct: {r}"
        )));
    }
    Ok((r, c))
}

/// Maximum length for a table name identifier.
/// SQLite has no identifier length limit, but we cap table names
/// to prevent pathological DDL strings from config input.
const MAX_IDENTIFIER_LEN: usize = 128;

/// Maximum identifier length accepted by `PostgreSQL`.
const POSTGRES_MAX_IDENTIFIER_LEN: usize = 63;

/// Maximum conversation table name length that leaves room for
/// `idx_` (4) and `_tenant_id` (10) in the generated index name.
const POSTGRES_MAX_CONVERSATION_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - 14;

/// Maximum items table name length that leaves room for `idx_` (4)
/// and `_conversation` (13) in the generated index name.
const POSTGRES_MAX_ITEMS_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - 17;

/// Maximum responses table name length that leaves room for the
/// `_schema_version` suffix in the derived version table name.
const POSTGRES_MAX_RESPONSES_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - SCHEMA_VERSION_SUFFIX.len();

/// Maximum responses table name length that leaves room for the
/// `_pending_approvals` suffix in the derived pending-approvals table
/// name. This suffix is longer than `_schema_version`, so it is the
/// binding constraint on the responses table name for `PostgreSQL`.
const POSTGRES_MAX_RESPONSES_TABLE_LEN_FOR_APPROVALS: usize =
    POSTGRES_MAX_IDENTIFIER_LEN - PENDING_APPROVALS_SUFFIX.len();

/// Reject identifiers that could cause SQL injection or invalid DDL.
pub(crate) fn validate_identifier(name: &str) -> Result<(), StoreError> {
    if name.is_empty() {
        return Err(StoreError::Database("table name must not be empty".to_owned()));
    }
    if name.len() > MAX_IDENTIFIER_LEN {
        return Err(StoreError::Database(format!(
            "table name exceeds {MAX_IDENTIFIER_LEN} characters: {name}"
        )));
    }
    if !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return Err(StoreError::Database(format!(
            "table name must start with a letter or underscore: {name}"
        )));
    }
    // Hyphens are valid in quoted SQLite identifiers but we
    // interpolate table names unquoted in SQL statements, so
    // restrict to alphanumeric + underscore to avoid quoting.
    if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err(StoreError::Database(format!(
            "table name contains invalid characters: {name}"
        )));
    }
    Ok(())
}

/// Validate the items table name and ensure it is distinct from the
/// responses and conversations tables.
fn validate_items_table<'a>(items: &'a str, responses: &str, conversations: &str) -> Result<&'a str, StoreError> {
    validate_identifier(items)?;
    if items.eq_ignore_ascii_case(responses) {
        return Err(StoreError::Database(format!(
            "items and response table names must be distinct: {items}"
        )));
    }
    if items.eq_ignore_ascii_case(conversations) {
        return Err(StoreError::Database(format!(
            "items and conversation table names must be distinct: {items}"
        )));
    }
    Ok(items)
}

/// Reject a `PostgreSQL` identifier that would be truncated.
fn validate_postgres_identifier_len(kind: &str, name: &str, max_len: usize) -> Result<(), StoreError> {
    if name.len() > max_len {
        return Err(StoreError::Database(format!(
            "{kind} exceeds PostgreSQL identifier limit of {max_len} bytes: {name}"
        )));
    }
    Ok(())
}

/// Reject a `PostgreSQL` identifier that would be case-folded.
///
/// DDL interpolates table names unquoted, so `PostgreSQL` folds them to
/// lowercase when creating the table. Schema validation then looks the name up
/// in `information_schema` as a bound parameter, which is compared literally
/// and is not folded. A mixed-case name therefore creates a lowercase table and
/// then fails to find it, reporting every expected column as missing.
///
/// Rejecting uppercase keeps the unquoted-interpolation design intact. The
/// alternative, quoting identifiers everywhere, would have to be applied
/// consistently across every DDL statement and query.
///
/// This is `PostgreSQL`-only. `SQLite` compares table names case-insensitively,
/// so a mixed-case name resolves to the same table on both paths there.
fn validate_postgres_identifier_case(kind: &str, name: &str) -> Result<(), StoreError> {
    if name.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(StoreError::Database(format!(
            "{kind} must be lowercase for PostgreSQL, which folds unquoted identifiers: {name}"
        )));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Expected Columns
// -----------------------------------------------------------------------------

/// Expected column names for the responses table.
pub(crate) const RESPONSES_COLUMNS: &[&str] = &[
    "tenant_id",
    "id",
    "created_at",
    "model",
    "response_object",
    "input",
    "messages",
];

/// Expected column names for the conversations table.
pub(crate) const CONVERSATIONS_COLUMNS: &[&str] =
    &["conversation_id", "tenant_id", "created_at", "metadata", "messages"];

/// Expected column names for the schema version table.
pub(crate) const VERSION_COLUMNS: &[&str] = &["version"];

/// Expected column names for the server-owned pending-approvals table.
pub(crate) const PENDING_APPROVALS_COLUMNS: &[&str] = &[
    "tenant_id",
    "response_id",
    "approval_id",
    "server_label",
    "tool_name",
    "arguments",
    "target_fingerprint",
    "created_at",
    "consumed_at",
];

/// Expected column names for the items table.
pub(crate) const ITEMS_COLUMNS: &[&str] = &[
    "item_id",
    "tenant_id",
    "conversation_id",
    "item_data",
    "created_at",
    "position",
];

/// Collect `(table_name, expected_columns)` pairs for schema validation.
pub(crate) fn expected_table_columns(tables: &TableNames) -> Vec<(&str, &'static [&'static str])> {
    let mut pairs = vec![
        (tables.responses.as_str(), RESPONSES_COLUMNS),
        (tables.conversations.as_str(), CONVERSATIONS_COLUMNS),
    ];
    if let Some(items) = &tables.items {
        pairs.push((items.as_str(), ITEMS_COLUMNS));
    }
    pairs
}

/// `(table_name, expected_columns, actual_columns)` triple for schema
/// validation.
pub(crate) type ColumnCheck<'a> = (&'a str, &'a [&'static str], &'a [String]);

/// Validate that every expected column exists in the actual column set.
///
/// Column name comparison is case-insensitive to handle backends that
/// fold identifiers.
///
/// # Errors
///
/// Returns [`StoreError::Database`] listing all tables with missing
/// columns.
pub(crate) fn check_column_presence(table_columns: &[ColumnCheck<'_>]) -> Result<(), StoreError> {
    let mut errors = Vec::new();
    for &(table, expected, actual) in table_columns {
        let missing: Vec<&str> = expected
            .iter()
            .filter(|col| !actual.iter().any(|a| a.eq_ignore_ascii_case(col)))
            .copied()
            .collect();
        if !missing.is_empty() {
            errors.push(format!("table '{table}' is missing columns: {}", missing.join(", ")));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(StoreError::Database(format!(
            "schema validation failed: {}",
            errors.join("; ")
        )))
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn valid_table_name() {
        validate_identifier("openai_responses").expect("valid name should pass");
    }

    #[test]
    fn valid_name_with_underscore_prefix() {
        validate_identifier("_internal").expect("underscore prefix should pass");
    }

    #[test]
    fn reject_empty_name() {
        let err = validate_identifier("").unwrap_err();
        assert!(err.to_string().contains("empty"), "should reject empty name: {err}");
    }

    #[test]
    fn reject_name_starting_with_digit() {
        let err = validate_identifier("123responses").unwrap_err();
        assert!(
            err.to_string().contains("start with"),
            "should reject digit prefix: {err}"
        );
    }

    #[test]
    fn reject_special_characters() {
        let err = validate_identifier("drop; DROP TABLE").unwrap_err();
        assert!(
            err.to_string().contains("invalid characters"),
            "should reject special chars: {err}"
        );
    }

    #[test]
    fn reject_hyphen() {
        let err = validate_identifier("my-table").unwrap_err();
        assert!(
            err.to_string().contains("invalid characters"),
            "should reject hyphen: {err}"
        );
    }

    #[test]
    fn reject_excessively_long_name() {
        let long = "a".repeat(MAX_IDENTIFIER_LEN + 1);
        let err = validate_identifier(&long).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "should reject long name: {err}");
    }

    #[test]
    fn generate_ddl_produces_valid_statements() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: None,
        };
        let ddl = generate_ddl(&tables).expect("valid names should produce DDL");
        assert_eq!(
            ddl.len(),
            5,
            "should produce 5 DDL statements (responses, conversations, tenant_id index, pending_approvals, version)"
        );
        assert!(
            ddl[0].contains("test_responses"),
            "first statement should reference responses table"
        );
    }

    #[test]
    fn generate_ddl_uses_bigint_for_created_at() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: None,
        };
        let ddl = generate_ddl(&tables).expect("valid names should produce DDL");

        assert!(
            ddl[0].contains("created_at      BIGINT NOT NULL"),
            "created_at should decode as i64 in Postgres: {}",
            ddl[0]
        );
    }

    #[test]
    fn generate_ddl_rejects_invalid_name() {
        let tables = TableNames {
            responses: "valid_name".to_owned(),
            conversations: "1invalid".to_owned(),
            items: None,
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("start with"),
            "should reject invalid conversation table name: {err}"
        );
    }

    #[test]
    fn generate_ddl_rejects_duplicate_names() {
        let tables = TableNames {
            responses: "same_table".to_owned(),
            conversations: "same_table".to_owned(),
            items: None,
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("distinct"),
            "should reject duplicate table names: {err}"
        );
    }

    #[test]
    fn generate_ddl_rejects_case_insensitive_duplicate_names() {
        let tables = TableNames {
            responses: "Responses".to_owned(),
            conversations: "responses".to_owned(),
            items: None,
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("distinct"),
            "should reject case-insensitive duplicate table names: {err}"
        );
    }

    #[test]
    fn postgres_identifier_rejects_truncated_table_name() {
        let tables = TableNames {
            responses: "r".repeat(POSTGRES_MAX_RESPONSES_TABLE_LEN + 1),
            conversations: "test_conversations".to_owned(),
            items: None,
        };
        let err = validate_postgres_identifiers(&tables).unwrap_err();

        assert!(
            err.to_string().contains("PostgreSQL identifier limit"),
            "should reject names PostgreSQL would truncate: {err}"
        );
    }

    #[test]
    fn postgres_identifier_rejects_truncated_index_name() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "c".repeat(POSTGRES_MAX_CONVERSATION_TABLE_LEN + 1),
            items: None,
        };
        let err = validate_postgres_identifiers(&tables).unwrap_err();

        assert!(
            err.to_string().contains("PostgreSQL identifier limit"),
            "should reject generated index names PostgreSQL would truncate: {err}"
        );
    }

    #[test]
    fn check_column_presence_passes_when_all_present() {
        let actual = vec!["id".to_owned(), "tenant_id".to_owned(), "created_at".to_owned()];
        let input = [("t", &["id", "tenant_id", "created_at"][..], actual.as_slice())];
        check_column_presence(&input).expect("all columns present should pass");
    }

    #[test]
    fn check_column_presence_detects_missing_columns() {
        let actual = vec!["id".to_owned(), "tenant_id".to_owned()];
        let input = [(
            "my_table",
            &["id", "tenant_id", "created_at", "model"][..],
            actual.as_slice(),
        )];
        let err = check_column_presence(&input).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("schema validation failed"), "{msg}");
        assert!(msg.contains("my_table"), "{msg}");
        assert!(msg.contains("created_at"), "{msg}");
        assert!(msg.contains("model"), "{msg}");
    }

    #[test]
    fn check_column_presence_is_case_insensitive() {
        let actual = vec!["ID".to_owned(), "TENANT_ID".to_owned()];
        let input = [("t", &["id", "tenant_id"][..], actual.as_slice())];
        check_column_presence(&input).expect("case-insensitive match should pass");
    }

    #[test]
    fn check_column_presence_aggregates_multiple_tables() {
        let actual_a = vec!["id".to_owned()];
        let actual_b = vec!["name".to_owned()];
        let input = [
            ("table_a", &["id", "missing_a"][..], actual_a.as_slice()),
            ("table_b", &["name", "missing_b"][..], actual_b.as_slice()),
        ];
        let err = check_column_presence(&input).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("table_a"), "{msg}");
        assert!(msg.contains("missing_a"), "{msg}");
        assert!(msg.contains("table_b"), "{msg}");
        assert!(msg.contains("missing_b"), "{msg}");
    }

    #[test]
    fn check_column_presence_tolerates_extra_columns() {
        let actual = vec!["id".to_owned(), "tenant_id".to_owned(), "extra_col".to_owned()];
        let input = [("t", &["id", "tenant_id"][..], actual.as_slice())];
        check_column_presence(&input).expect("extra columns should not cause failure");
    }

    #[test]
    fn expected_table_columns_excludes_items_when_none() {
        let tables = TableNames {
            responses: "r".to_owned(),
            conversations: "c".to_owned(),
            items: None,
        };
        assert_eq!(
            expected_table_columns(&tables).len(),
            2,
            "should have responses and conversations only"
        );
    }

    #[test]
    fn expected_table_columns_includes_items_when_configured() {
        let tables = TableNames {
            responses: "r".to_owned(),
            conversations: "c".to_owned(),
            items: Some("i".to_owned()),
        };
        let pairs = expected_table_columns(&tables);
        assert_eq!(pairs.len(), 3, "should include items table");
        assert_eq!(pairs[2].0, "i", "third pair should be the items table");
        assert_eq!(pairs[2].1, ITEMS_COLUMNS, "items columns should match");
    }

    #[test]
    fn schema_version_table_derives_name() {
        assert_eq!(
            schema_version_table("openai_responses"),
            "openai_responses_schema_version"
        );
    }

    #[test]
    fn generate_ddl_includes_version_table() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: None,
        };
        let ddl = generate_ddl(&tables).expect("valid names should produce DDL");
        let version_ddl = ddl.last().expect("should have statements");
        assert!(
            version_ddl.contains("test_responses_schema_version"),
            "last DDL should create version table: {version_ddl}"
        );
        assert!(
            version_ddl.contains("version BIGINT NOT NULL"),
            "version table should have version column: {version_ddl}"
        );
    }

    #[test]
    fn pending_approvals_table_derives_name() {
        assert_eq!(
            pending_approvals_table("openai_responses"),
            "openai_responses_pending_approvals"
        );
    }

    #[test]
    fn generate_ddl_includes_pending_approvals_table() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: None,
        };
        let ddl = generate_ddl(&tables).expect("valid names should produce DDL");
        // The pending-approvals table is created just before the version
        // table, which must remain last.
        let approvals_ddl = &ddl[ddl.len() - 2];
        assert!(
            approvals_ddl.contains("test_responses_pending_approvals"),
            "second-to-last DDL should create the pending-approvals table: {approvals_ddl}"
        );
        // The issuing response_id scopes every approval, and the primary key
        // binds each single-use token to (tenant_id, response_id, approval_id).
        for expected in [
            "response_id        TEXT NOT NULL",
            "approval_id        TEXT NOT NULL",
            "target_fingerprint TEXT NOT NULL",
            "consumed_at        BIGINT",
            "PRIMARY KEY (tenant_id, response_id, approval_id)",
        ] {
            assert!(
                approvals_ddl.contains(expected),
                "pending-approvals DDL should contain `{expected}`: {approvals_ddl}"
            );
        }
    }

    #[test]
    fn generate_ddl_rejects_pending_approvals_collision_with_conversations() {
        let tables = TableNames {
            responses: "test".to_owned(),
            conversations: "test_pending_approvals".to_owned(),
            items: None,
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("collides with conversation table"),
            "should reject collision: {err}"
        );
    }

    #[test]
    fn generate_ddl_rejects_pending_approvals_collision_with_items() {
        let tables = TableNames {
            responses: "test".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: Some("test_pending_approvals".to_owned()),
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("collides with items table"),
            "should reject collision: {err}"
        );
    }

    #[test]
    fn postgres_identifier_rejects_long_responses_for_pending_approvals_table() {
        // A responses name that fits the version-table suffix but not the
        // longer pending-approvals suffix must still be rejected.
        let responses = "r".repeat(POSTGRES_MAX_RESPONSES_TABLE_LEN_FOR_APPROVALS + 1);
        assert!(
            responses.len() <= POSTGRES_MAX_RESPONSES_TABLE_LEN,
            "test premise: name should fit the shorter version suffix"
        );
        let tables = TableNames {
            responses,
            conversations: "c".to_owned(),
            items: None,
        };
        let err = validate_postgres_identifiers(&tables).unwrap_err();
        assert!(
            err.to_string().contains("PostgreSQL identifier limit"),
            "should reject responses name that makes the pending-approvals table too long: {err}"
        );
    }

    #[test]
    fn postgres_identifier_rejects_long_responses_for_version_table() {
        let tables = TableNames {
            responses: "r".repeat(POSTGRES_MAX_RESPONSES_TABLE_LEN + 1),
            conversations: "c".to_owned(),
            items: None,
        };
        let err = validate_postgres_identifiers(&tables).unwrap_err();
        assert!(
            err.to_string().contains("PostgreSQL identifier limit"),
            "should reject responses name that makes version table too long: {err}"
        );
    }

    #[test]
    fn generate_ddl_rejects_version_table_collision_with_conversations() {
        let tables = TableNames {
            responses: "test".to_owned(),
            conversations: "test_schema_version".to_owned(),
            items: None,
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("collides with conversation table"),
            "should reject collision: {err}"
        );
    }

    #[test]
    fn generate_ddl_rejects_version_table_collision_with_items() {
        let tables = TableNames {
            responses: "test".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: Some("test_schema_version".to_owned()),
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string().contains("collides with items table"),
            "should reject collision: {err}"
        );
    }

    #[test]
    fn generate_ddl_includes_items_ddl() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: Some("test_items".to_owned()),
        };
        let ddl = generate_ddl(&tables).expect("valid names with items should produce DDL");
        assert_eq!(
            ddl.len(),
            8,
            "should produce 8 DDL statements (responses, conversations, tenant_id index, items, items indexes, \
             pending_approvals, version)"
        );
        assert!(
            ddl[3].contains("test_items"),
            "fourth statement should create items table: {}",
            ddl[3]
        );
        assert!(
            ddl[4].contains("idx_test_items_conversation"),
            "fifth statement should create items index: {}",
            ddl[4]
        );
    }

    #[test]
    fn generate_ddl_rejects_items_same_as_responses() {
        let tables = TableNames {
            responses: "shared_name".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: Some("shared_name".to_owned()),
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string()
                .contains("items and response table names must be distinct"),
            "should reject items == responses: {err}"
        );
    }

    #[test]
    fn generate_ddl_rejects_items_same_as_conversations() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "shared_name".to_owned(),
            items: Some("shared_name".to_owned()),
        };
        let err = generate_ddl(&tables).unwrap_err();
        assert!(
            err.to_string()
                .contains("items and conversation table names must be distinct"),
            "should reject items == conversations: {err}"
        );
    }

    #[test]
    fn validate_postgres_table_identifiers_accepts_valid() {
        validate_postgres_table_identifiers("test_responses", "test_conversations")
            .expect("valid names should pass PostgreSQL validation");
    }

    #[test]
    fn validate_postgres_table_set_identifiers_accepts_valid_with_items() {
        validate_postgres_table_set_identifiers("test_responses", "test_conversations", Some("test_items"))
            .expect("valid names with items should pass PostgreSQL validation");
    }

    #[test]
    fn validate_postgres_table_set_identifiers_rejects_long_items() {
        let err = validate_postgres_table_set_identifiers(
            "test_responses",
            "test_conversations",
            Some(&"i".repeat(POSTGRES_MAX_ITEMS_TABLE_LEN + 1)),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("PostgreSQL identifier limit"),
            "should reject items name PostgreSQL would truncate: {err}"
        );
    }

    #[test]
    fn postgres_rejects_uppercase_responses_table() {
        let err = validate_postgres_table_identifiers("OpenAIResponses", "test_conversations").unwrap_err();
        assert!(
            err.to_string().contains("must be lowercase"),
            "should reject a name PostgreSQL would case-fold: {err}"
        );
        assert!(
            err.to_string().contains("response table name"),
            "error should name the offending field: {err}"
        );
    }

    #[test]
    fn postgres_rejects_uppercase_conversations_table() {
        let err = validate_postgres_table_identifiers("test_responses", "OpenAIConversations").unwrap_err();
        assert!(
            err.to_string().contains("must be lowercase"),
            "should reject a name PostgreSQL would case-fold: {err}"
        );
        assert!(
            err.to_string().contains("conversation table name"),
            "error should name the offending field: {err}"
        );
    }

    #[test]
    fn postgres_rejects_uppercase_items_table() {
        let err =
            validate_postgres_table_set_identifiers("test_responses", "test_conversations", Some("ConversationItems"))
                .unwrap_err();
        assert!(
            err.to_string().contains("must be lowercase"),
            "should reject a name PostgreSQL would case-fold: {err}"
        );
        assert!(
            err.to_string().contains("items table name"),
            "error should name the offending field: {err}"
        );
    }

    #[test]
    fn postgres_rejects_a_single_uppercase_character() {
        let err = validate_postgres_table_identifiers("test_responseS", "test_conversations").unwrap_err();
        assert!(
            err.to_string().contains("must be lowercase"),
            "the DDL folds the whole identifier, so one uppercase byte is enough to make the created \
             table name differ from the configured one: {err}"
        );
    }

    #[test]
    fn postgres_accepts_digits_and_underscores() {
        validate_postgres_table_identifiers("responses_v2", "_conversations_2026")
            .expect("lowercase names with digits and underscores should pass");
    }

    #[test]
    fn shared_identifier_validation_stays_case_permissive() {
        validate_identifier("OpenAIResponses")
            .expect("SQLite compares table names case-insensitively, so only the PostgreSQL path rejects uppercase");
    }
}
