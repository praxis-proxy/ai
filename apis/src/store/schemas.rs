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
///
/// Version 3 stores the responses table's JSON payload columns
/// (`response_object`, `input`, `messages`) as native binary (`BLOB`
/// on SQLite, `BYTEA` on `PostgreSQL`) instead of `TEXT` so the
/// response store can persist compressed payloads. Version 2 databases
/// must be migrated before use.
pub(crate) const SCHEMA_VERSION: i64 = 3;

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
// SQL Dialect
// -----------------------------------------------------------------------------

/// SQL dialect a store targets, used to pick dialect-specific types in
/// generated DDL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SqlDialect {
    /// SQLite backend.
    #[cfg(feature = "store-sqlite")]
    Sqlite,
    /// `PostgreSQL` backend.
    #[cfg(feature = "store-postgres")]
    Postgres,
}

impl SqlDialect {
    /// Column type for a binary JSON payload column.
    fn bytes_type(self) -> &'static str {
        match self {
            #[cfg(feature = "store-sqlite")]
            SqlDialect::Sqlite => "BLOB",
            #[cfg(feature = "store-postgres")]
            SqlDialect::Postgres => "BYTEA",
        }
    }
}

// -----------------------------------------------------------------------------
// Schema DDL
// -----------------------------------------------------------------------------

/// Generate DDL statements for the given table names.
///
/// Each statement uses `IF NOT EXISTS` so it is safe to run on
/// every startup. The schema uses TEXT for JSON columns (standard
/// `SQLite` pattern) with the exception of the responses table which
/// uses the dialect's binary type (`BLOB` on SQLite, `BYTEA` on `PostgreSQL`)
/// so the responses store can persist compressed payloads; Timestamps
/// use BIGINT so the same DDL is compatible with `PostgreSQL` `i64` decoding.
///
/// # Errors
///
/// Returns [`StoreError::Database`] if table names contain
/// invalid characters.
#[expect(clippy::too_many_lines, reason = "linear DDL statement assembly per table")]
pub(crate) fn generate_ddl(tables: &TableNames, dialect: SqlDialect) -> Result<Vec<String>, StoreError> {
    let (r, c) = validate_table_names(tables)?;
    let bytes_type = dialect.bytes_type();

    let mut stmts = vec![
        responses_ddl(r, bytes_type),
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
#[cfg(feature = "store-postgres")]
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
#[cfg(feature = "store-postgres")]
pub(crate) fn validate_postgres_table_identifiers(
    responses_table: &str,
    conversations_table: &str,
) -> Result<(), StoreError> {
    validate_postgres_table_set_identifiers(responses_table, conversations_table, None)
}

/// Validate table identifiers for a store that may also configure
/// conversation item rows.
#[cfg(feature = "store-postgres")]
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
fn responses_ddl(r: &str, bytes_type: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {r} (
            id              TEXT NOT NULL,
            tenant_id       TEXT NOT NULL,
            owner_issuer    TEXT NOT NULL,
            owner_subject   TEXT NOT NULL,
            created_at      BIGINT NOT NULL,
            model           TEXT NOT NULL,
            response_object {bytes_type} NOT NULL,
            input           {bytes_type} NOT NULL,
            messages        {bytes_type} NOT NULL,
            PRIMARY KEY (id)
        )"
    )
}

/// DDL for the conversations table.
fn conversations_ddl(c: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {c} (
            conversation_id TEXT NOT NULL,
            tenant_id       TEXT NOT NULL,
            owner_issuer    TEXT NOT NULL,
            owner_subject   TEXT NOT NULL,
            created_at      BIGINT NOT NULL,
            metadata        TEXT NOT NULL,
            messages        TEXT NOT NULL,
            PRIMARY KEY (conversation_id)
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
            owner_issuer      TEXT NOT NULL,
            owner_subject     TEXT NOT NULL,
            conversation_id   TEXT NOT NULL,
            item_data         TEXT NOT NULL,
            created_at        BIGINT NOT NULL,
            position          BIGINT NOT NULL,
            PRIMARY KEY (item_id)
        )"
    ));
    stmts.push(format!(
        "CREATE INDEX IF NOT EXISTS idx_{i}_conversation \
         ON {i}(conversation_id, position, item_id)"
    ));
    stmts.push(format!(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_{i}_position \
         ON {i}(conversation_id, position)"
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
            owner_issuer       TEXT NOT NULL,
            owner_subject      TEXT NOT NULL,
            response_id        TEXT NOT NULL,
            approval_id        TEXT NOT NULL,
            server_label       TEXT NOT NULL,
            tool_name          TEXT NOT NULL,
            arguments          TEXT NOT NULL,
            target_fingerprint TEXT NOT NULL,
            created_at         BIGINT NOT NULL,
            consumed_at        BIGINT,
            PRIMARY KEY (response_id, approval_id)
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
#[cfg(feature = "store-postgres")]
const POSTGRES_MAX_IDENTIFIER_LEN: usize = 63;

/// Maximum conversation table name length that leaves room for
/// `idx_` (4) and `_tenant_id` (10) in the generated index name.
#[cfg(feature = "store-postgres")]
const POSTGRES_MAX_CONVERSATION_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - 14;

/// Maximum items table name length that leaves room for `idx_` (4)
/// and `_conversation` (13) in the generated index name.
#[cfg(feature = "store-postgres")]
const POSTGRES_MAX_ITEMS_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - 17;

/// Maximum responses table name length that leaves room for the
/// `_schema_version` suffix in the derived version table name.
#[cfg(feature = "store-postgres")]
const POSTGRES_MAX_RESPONSES_TABLE_LEN: usize = POSTGRES_MAX_IDENTIFIER_LEN - SCHEMA_VERSION_SUFFIX.len();

/// Maximum responses table name length that leaves room for the
/// `_pending_approvals` suffix in the derived pending-approvals table
/// name. This suffix is longer than `_schema_version`, so it is the
/// binding constraint on the responses table name for `PostgreSQL`.
#[cfg(feature = "store-postgres")]
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
#[cfg(feature = "store-postgres")]
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
#[cfg(feature = "store-postgres")]
fn validate_postgres_identifier_case(kind: &str, name: &str) -> Result<(), StoreError> {
    if name.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(StoreError::Database(format!(
            "{kind} must be lowercase for PostgreSQL, which folds unquoted identifiers: {name}"
        )));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Expected Schema Contract
// -----------------------------------------------------------------------------

/// Expected column names for the responses table.
const RESPONSES_COLUMNS: &[&str] = &[
    "tenant_id",
    "id",
    "owner_issuer",
    "owner_subject",
    "created_at",
    "model",
    "response_object",
    "input",
    "messages",
];

/// Expected column names for the conversations table.
pub(crate) const CONVERSATIONS_COLUMNS: &[&str] = &[
    "conversation_id",
    "tenant_id",
    "owner_issuer",
    "owner_subject",
    "created_at",
    "metadata",
    "messages",
];

/// Expected column names for the server-owned pending-approvals table.
pub(crate) const PENDING_APPROVALS_COLUMNS: &[&str] = &[
    "tenant_id",
    "owner_issuer",
    "owner_subject",
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
const ITEMS_COLUMNS: &[&str] = &[
    "item_id",
    "tenant_id",
    "owner_issuer",
    "owner_subject",
    "conversation_id",
    "item_data",
    "created_at",
    "position",
];

/// Expected ordered primary key columns for the responses table.
///
/// Response IDs are globally owner-immutable. Store writes use this conflict
/// target and reject an existing row whose complete owner does not match.
const RESPONSES_PRIMARY_KEY: &[&str] = &["id"];

/// Expected ordered primary key columns for the conversations table.
const CONVERSATIONS_PRIMARY_KEY: &[&str] = &["conversation_id"];

/// Expected ordered primary key columns for the items table.
const ITEMS_PRIMARY_KEY: &[&str] = &["item_id"];

/// Columns of the one unique index the items table generates beyond its primary
/// key (`idx_<items>_position`).
///
/// Conversation IDs are globally owner-immutable, so they safely scope item
/// positions without repeating identity columns in the index.
const ITEMS_POSITION_UNIQUE: &[&str] = &["conversation_id", "position"];

/// Expected ordered primary key columns for the server-owned pending-approvals
/// table.
///
/// Approval rows inherit their issuing Response owner. Since Response IDs are
/// globally owner-immutable, `(response_id, approval_id)` is the conflict key.
const PENDING_APPROVALS_PRIMARY_KEY: &[&str] = &["response_id", "approval_id"];

/// The schema this store generates for one table: the columns it must contain
/// and its exact ordered primary key.
///
/// Validation is fail-closed against this contract. Rather than proving a
/// pre-existing schema is a safe superset of ours, we require it to match what
/// we generate; any deviation -- an unknown type, a folding collation, an
/// unexpected unique index, or a reordered key -- is rejected.
#[derive(Clone, Copy)]
pub(crate) struct ExpectedTable {
    /// Columns that must be present. Extra columns are tolerated.
    pub columns: &'static [&'static str],
    /// The exact ordered primary key columns.
    pub primary_key: &'static [&'static str],
    /// Column sets of the unique indexes this store generates beyond the primary
    /// key. Each entry is one unique index's columns. A discovered unique index
    /// is accepted only when its column set matches one of these; any other
    /// unique index is rejected because it changes the store's collision
    /// semantics. Empty for tables that generate no unique index beyond the
    /// primary key.
    pub unique_indexes: &'static [&'static [&'static str]],
}

/// The responses table contract.
const RESPONSES_TABLE: ExpectedTable = ExpectedTable {
    columns: RESPONSES_COLUMNS,
    primary_key: RESPONSES_PRIMARY_KEY,
    unique_indexes: &[],
};

/// The conversations table contract.
const CONVERSATIONS_TABLE: ExpectedTable = ExpectedTable {
    columns: CONVERSATIONS_COLUMNS,
    primary_key: CONVERSATIONS_PRIMARY_KEY,
    unique_indexes: &[],
};

/// The items table contract.
const ITEMS_TABLE: ExpectedTable = ExpectedTable {
    columns: ITEMS_COLUMNS,
    primary_key: ITEMS_PRIMARY_KEY,
    unique_indexes: &[ITEMS_POSITION_UNIQUE],
};

/// The server-owned pending-approvals table contract.
const PENDING_APPROVALS_TABLE: ExpectedTable = ExpectedTable {
    columns: PENDING_APPROVALS_COLUMNS,
    primary_key: PENDING_APPROVALS_PRIMARY_KEY,
    unique_indexes: &[],
};

/// Collect the `(table_name, expected)` contract for every owner-scoped table.
///
/// The server-owned pending-approvals table is included because its rows inherit
/// Response ownership. The single-row schema version table holds no owner data;
/// it is validated by value in `check_schema_version`, not structurally.
///
/// Table names are returned owned because the pending-approvals name is derived
/// from the responses name rather than borrowed from `tables`.
pub(crate) fn expected_tables(tables: &TableNames) -> Vec<(String, ExpectedTable)> {
    let mut expected = vec![
        (tables.responses.clone(), RESPONSES_TABLE),
        (tables.conversations.clone(), CONVERSATIONS_TABLE),
    ];
    if let Some(items) = &tables.items {
        expected.push((items.clone(), ITEMS_TABLE));
    }
    expected.push((pending_approvals_table(&tables.responses), PENDING_APPROVALS_TABLE));
    expected
}

// -----------------------------------------------------------------------------
// Discovered Schema
// -----------------------------------------------------------------------------

/// A table's schema as discovered from the backend catalog, normalized so the
/// comparison against [`ExpectedTable`] is backend-agnostic.
#[derive(Debug)]
pub(crate) struct ActualTable {
    /// Column names present on the table.
    pub columns: Vec<String>,
    /// The ordered primary key columns, each carrying a folding verdict.
    pub primary_key: Vec<ActualKeyColumn>,
    /// Whether the primary key is enforced immediately (not `DEFERRABLE`).
    /// `SQLite` has no deferrable keys and always reports `true`; `PostgreSQL`
    /// reads `pg_index.indimmediate`. A deferrable key over the conflict target
    /// cannot arbitrate an `ON CONFLICT` upsert.
    pub primary_key_immediate: bool,
    /// Unique indexes other than the primary key, each with the columns it
    /// covers. An index is accepted only when its column set matches one the
    /// store generates (see [`ExpectedTable::unique_indexes`]); any other unique
    /// index is rejected because it changes the store's collision semantics.
    pub unique_indexes: Vec<ActualUniqueIndex>,
}

/// A unique index discovered from the catalog, other than the primary key.
#[derive(Debug)]
pub(crate) struct ActualUniqueIndex {
    /// Index name, used only in diagnostics.
    pub name: String,
    /// The columns the index covers, in catalog order.
    pub columns: Vec<String>,
}

/// A single primary key column as discovered from the catalog.
#[derive(Debug)]
pub(crate) struct ActualKeyColumn {
    /// Column name.
    pub name: String,
    /// A human-readable reason the column folds distinct key values together --
    /// a wrong type or affinity, a folding collation, or an untrusted operator
    /// class -- or `None` when the column preserves distinctness. Each backend
    /// computes this at discovery via its key-column folding check so the
    /// comparison stays backend-agnostic.
    pub folding: Option<String>,
}

/// `(table_name, expected, actual)` triple for the schema comparison.
pub(crate) type SchemaCheck<'a> = (&'a str, ExpectedTable, &'a ActualTable);

/// Validate every discovered table against the schema this store generates.
///
/// This is the single fail-closed comparison behind startup validation. A table
/// passes only when it has every expected column, exactly the expected ordered
/// primary key, a distinctness-preserving type, collation, and operator class on
/// each key column, an immediate primary key, and no unique index beyond the
/// primary key and the store's own generated unique indexes. Anything else fails
/// closed and requires a migration, because `CREATE TABLE IF NOT EXISTS`
/// preserves a pre-existing table that could silently lose data across tenants.
///
/// # Errors
///
/// Returns [`StoreError::Database`] listing every deviation across all tables.
pub(crate) fn check_schema(checks: &[SchemaCheck<'_>]) -> Result<(), StoreError> {
    let mut errors = Vec::new();
    for &(table, expected, actual) in checks {
        check_columns(table, expected.columns, &actual.columns, &mut errors);
        check_primary_key(table, expected.primary_key, actual, &mut errors);
        check_unique_indexes(table, expected.unique_indexes, &actual.unique_indexes, &mut errors);
    }
    into_validation_result(&errors)
}

/// Record any expected column missing from the discovered set.
///
/// Comparison is case-insensitive to tolerate backends that fold identifiers;
/// extra columns are tolerated.
fn check_columns(table: &str, expected: &[&str], actual: &[String], errors: &mut Vec<String>) {
    let missing: Vec<&str> = expected
        .iter()
        .filter(|col| !actual.iter().any(|a| a.eq_ignore_ascii_case(col)))
        .copied()
        .collect();
    if !missing.is_empty() {
        errors.push(format!("table '{table}' is missing columns: {}", missing.join(", ")));
    }
}

/// Record a primary key that is not the exact ordered contract, a key column
/// that folds distinct values, or a deferrable key.
///
/// Comparison is order-sensitive to match the generated DDL and case-insensitive
/// per column to tolerate backends that fold identifiers.
fn check_primary_key(table: &str, expected: &[&str], actual: &ActualTable, errors: &mut Vec<String>) {
    let actual_columns: Vec<&str> = actual.primary_key.iter().map(|column| column.name.as_str()).collect();
    let matches = expected.len() == actual_columns.len()
        && expected
            .iter()
            .zip(&actual_columns)
            .all(|(expected_col, actual_col)| actual_col.eq_ignore_ascii_case(expected_col));
    if !matches {
        errors.push(format!(
            "table '{table}' has primary key ({}), expected ({})",
            actual_columns.join(", "),
            expected.join(", ")
        ));
    }
    for column in &actual.primary_key {
        if let Some(reason) = &column.folding {
            errors.push(format!("table '{table}' primary key column '{}' {reason}", column.name));
        }
    }
    if !actual.primary_key_immediate {
        errors.push(format!(
            "table '{table}' has a deferrable primary key; ON CONFLICT upserts require an immediate primary key"
        ));
    }
}

/// Record any unique index whose column set is not one the store generates.
///
/// The primary key is validated separately and excluded upstream, so every entry
/// here is a unique index *beyond* the primary key. Each is accepted only when
/// its columns match one of the expected sets; matching is case-insensitive and
/// order-insensitive because uniqueness is a property of the column set, not its
/// order. Any unmatched unique index fails closed because it changes the
/// collision behavior of `INSERT OR REPLACE` / `ON CONFLICT`.
fn check_unique_indexes(table: &str, expected: &[&[&str]], actual: &[ActualUniqueIndex], errors: &mut Vec<String>) {
    for index in actual {
        let matches = expected.iter().any(|columns| same_column_set(columns, &index.columns));
        if !matches {
            errors.push(format!(
                "table '{table}' has unexpected unique index '{}' on ({}); \
                 only the primary key and the store's own unique indexes may be unique",
                index.name,
                index.columns.join(", ")
            ));
        }
    }
}

/// Whether two column lists denote the same set, case-insensitively and
/// independent of order.
fn same_column_set(expected: &[&str], actual: &[String]) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .all(|col| actual.iter().any(|a| a.eq_ignore_ascii_case(col)))
}

/// Collapse accumulated schema-validation errors into a single [`Result`].
///
/// An empty list is success; otherwise every offending item is joined into one
/// [`StoreError::Database`] with the shared "migration required" envelope so
/// callers surface all problems at once.
fn into_validation_result(errors: &[String]) -> Result<(), StoreError> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(StoreError::Database(format!(
            "schema validation failed: {}; database migration required",
            errors.join("; ")
        )))
    }
}

// -----------------------------------------------------------------------------
// SQLite Key-Column Folding
// -----------------------------------------------------------------------------

/// `SQLite`'s default collation and the only one guaranteed to keep distinct
/// text keys distinct. `NOCASE` folds ASCII case and `RTRIM` folds trailing
/// spaces, so either would let two distinct tenant or response ids compare
/// equal and collapse under `INSERT OR REPLACE`.
#[cfg(feature = "store-sqlite")]
const SQLITE_SAFE_COLLATION: &str = "BINARY";

/// Whether a `SQLite` collation folds distinct text values together.
///
/// Only `BINARY` (the default) is guaranteed value-preserving. `NOCASE`,
/// `RTRIM`, and any custom sequence may compare two different strings equal, so
/// they are treated as folding.
#[cfg(feature = "store-sqlite")]
pub(crate) fn sqlite_collation_folds(collation: &str) -> bool {
    !collation.eq_ignore_ascii_case(SQLITE_SAFE_COLLATION)
}

/// A `SQLite` type affinity class, derived from a column's declared type.
#[derive(Debug, PartialEq, Eq)]
#[cfg(feature = "store-sqlite")]
enum SqliteAffinity {
    /// Declared type contains `INT`; stores integers and coerces numeric text.
    Integer,
    /// Declared type contains `CHAR`, `CLOB`, or `TEXT`; the safe key affinity.
    Text,
    /// Declared type contains `BLOB` or is empty; stores data as-is.
    Blob,
    /// Declared type contains `REAL`, `FLOA`, or `DOUB`; stores floats.
    Real,
    /// Fallback affinity; coerces numeric-looking text to integer or real.
    Numeric,
}

#[cfg(feature = "store-sqlite")]
impl SqliteAffinity {
    /// The affinity's canonical name for diagnostics.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Integer => "INTEGER",
            Self::Text => "TEXT",
            Self::Blob => "BLOB",
            Self::Real => "REAL",
            Self::Numeric => "NUMERIC",
        }
    }
}

/// Determine a declared column type's `SQLite` affinity, following the rules
/// at <https://www.sqlite.org/datatype3.html#determination_of_column_affinity>.
///
/// The rules are ordered: `INT` implies INTEGER, and `CHAR`/`CLOB`/`TEXT` imply
/// TEXT. This is why `VARCHAR(255)` resolves to TEXT while `BIGINT` resolves to
/// INTEGER.
#[cfg(feature = "store-sqlite")]
fn sqlite_type_affinity(declared_type: &str) -> SqliteAffinity {
    let upper = declared_type.to_ascii_uppercase();
    if upper.contains("INT") {
        SqliteAffinity::Integer
    } else if upper.contains("CHAR") || upper.contains("CLOB") || upper.contains("TEXT") {
        SqliteAffinity::Text
    } else if upper.contains("BLOB") || upper.is_empty() {
        SqliteAffinity::Blob
    } else if upper.contains("REAL") || upper.contains("FLOA") || upper.contains("DOUB") {
        SqliteAffinity::Real
    } else {
        SqliteAffinity::Numeric
    }
}

/// Folding verdict for a `SQLite` primary key column, from its declared type and
/// effective collation.
///
/// A key column preserves distinctness only when it has TEXT affinity and a
/// value-preserving collation. A numeric affinity coerces `"1"` and `"01"` to
/// the same integer, and a folding collation (`NOCASE`, `RTRIM`, ...) compares
/// distinct strings equal, so either collapses two distinct ids under
/// `INSERT OR REPLACE`. Affinity is derived from the declared type because it is
/// invisible to the index pragmas; the collation is read from the primary key's
/// backing index.
#[cfg(feature = "store-sqlite")]
pub(crate) fn sqlite_key_column_folding(declared_type: &str, collation: Option<&str>) -> Option<String> {
    let affinity = sqlite_type_affinity(declared_type);
    if affinity != SqliteAffinity::Text {
        return Some(format!(
            "has type '{declared_type}' with {} affinity, expected TEXT affinity",
            affinity.as_str()
        ));
    }
    if let Some(collation) = collation.filter(|c| sqlite_collation_folds(c)) {
        return Some(format!(
            "uses collation '{collation}' that folds distinct values; a value-preserving collation is required"
        ));
    }
    None
}

// -----------------------------------------------------------------------------
// PostgreSQL Key-Column Folding
// -----------------------------------------------------------------------------

/// Stable `pg_catalog` type OIDs whose equality is byte-exact under a
/// deterministic collation, so they keep distinct tenant or response ids
/// distinct: `text` (25) and `varchar` (1043). Both are assigned fixed OIDs in
/// `PostgreSQL`'s bootstrap catalog, so matching by OID is version-independent
/// and cannot be spoofed by a same-named type in another schema. `bpchar` (1042)
/// is deliberately excluded because it blank-pads, folding `'a'` and `'a '`;
/// every other type -- `citext`, an enum, a custom type, or a `DOMAIN` (whose
/// column reports the domain's own OID, never one of these) -- is likewise
/// rejected.
#[cfg(feature = "store-postgres")]
pub(crate) const PG_ALLOWED_KEY_TYPE_OIDS: &[i64] = &[25, 1043];

/// Folding verdict for a `PostgreSQL` primary key column, from its catalog
/// metadata.
///
/// A key column preserves distinctness only when all three hold: its type is a
/// trusted built-in text type (allow-listed by the stable bootstrap OIDs in
/// [`PG_ALLOWED_KEY_TYPE_OIDS`], so `citext`, `bpchar`, an enum, or a `DOMAIN` --
/// which reports its own OID -- all fail closed with no recursive type walk);
/// its collation is deterministic (a non-deterministic collation makes distinct
/// strings equal for the `ON CONFLICT` arbiter); and its operator class is the
/// built-in `text_ops` B-tree class that backs both `text` and `varchar` keys
/// (validated structurally by the discovery query, never by a spoofable name).
/// The type is checked first, so a
/// non-collatable column -- whose `collation_deterministic` is `None` -- can only
/// reach the later checks with an allow-listed type, which is always collatable.
#[cfg(feature = "store-postgres")]
pub(crate) fn pg_key_column_folding(
    type_oid: i64,
    type_name: &str,
    collation_deterministic: Option<bool>,
    operator_class_trusted: bool,
) -> Option<String> {
    if !PG_ALLOWED_KEY_TYPE_OIDS.contains(&type_oid) {
        return Some(format!(
            "has type '{type_name}' (OID {type_oid}); only the built-in text or varchar types are supported"
        ));
    }
    if collation_deterministic == Some(false) {
        return Some(
            "uses a non-deterministic collation that folds distinct values; a deterministic collation is required"
                .to_owned(),
        );
    }
    if !operator_class_trusted {
        return Some(
            "uses a non-default operator class that may fold distinct values; the built-in B-tree operator class is required"
                .to_owned(),
        );
    }
    None
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[cfg(all(feature = "store-postgres", feature = "store-sqlite"))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::type_complexity,
    reason = "tests"
)]
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
        let ddl = generate_ddl(&tables, SqlDialect::Sqlite).expect("valid names should produce DDL");
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
        let ddl = generate_ddl(&tables, SqlDialect::Sqlite).expect("valid names should produce DDL");

        assert!(
            ddl[0].contains("created_at      BIGINT NOT NULL"),
            "created_at should decode as i64 in Postgres: {}",
            ddl[0]
        );
    }

    #[test]
    fn generate_ddl_sqlite_uses_blob_for_responses_payload_columns() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: Some("test_items".to_owned()),
        };
        let ddl = generate_ddl(&tables, SqlDialect::Sqlite).expect("valid names should produce DDL");
        assert!(
            ddl[0].contains("response_object BLOB NOT NULL"),
            "SQLite responses payload should be BLOB: {}",
            ddl[0]
        );
        assert!(ddl[0].contains("input           BLOB NOT NULL"), "{}", ddl[0]);
        assert!(ddl[0].contains("messages        BLOB NOT NULL"), "{}", ddl[0]);
        assert!(!ddl[0].contains("BYTEA"), "SQLite must not use BYTEA: {}", ddl[0]);
        // Conversations and items stay TEXT.
        assert!(ddl[1].contains("messages        TEXT NOT NULL"), "{}", ddl[1]);
        assert!(ddl[3].contains("item_data         TEXT NOT NULL"), "{}", ddl[3]);
    }

    #[test]
    fn generate_ddl_postgres_uses_bytea_for_responses_payload_columns() {
        let tables = TableNames {
            responses: "test_responses".to_owned(),
            conversations: "test_conversations".to_owned(),
            items: Some("test_items".to_owned()),
        };
        let ddl = generate_ddl(&tables, SqlDialect::Postgres).expect("valid names should produce DDL");
        assert!(
            ddl[0].contains("response_object BYTEA NOT NULL"),
            "Postgres responses payload should be BYTEA: {}",
            ddl[0]
        );
        assert!(ddl[0].contains("input           BYTEA NOT NULL"), "{}", ddl[0]);
        assert!(ddl[0].contains("messages        BYTEA NOT NULL"), "{}", ddl[0]);
        assert!(!ddl[0].contains("BLOB"), "Postgres must not use BLOB: {}", ddl[0]);
        // Conversations and items stay TEXT.
        assert!(ddl[1].contains("messages        TEXT NOT NULL"), "{}", ddl[1]);
        assert!(ddl[3].contains("item_data         TEXT NOT NULL"), "{}", ddl[3]);
    }

    #[test]
    fn generate_ddl_rejects_invalid_name() {
        let tables = TableNames {
            responses: "valid_name".to_owned(),
            conversations: "1invalid".to_owned(),
            items: None,
        };
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
    fn expected_tables_excludes_items_when_none() {
        let tables = TableNames {
            responses: "r".to_owned(),
            conversations: "c".to_owned(),
            items: None,
        };
        let expected = expected_tables(&tables);
        assert_eq!(
            expected.len(),
            3,
            "should have responses, conversations, and pending-approvals only"
        );
        assert_eq!(
            expected[2].0, "r_pending_approvals",
            "last entry should be the derived pending-approvals table"
        );
        assert_eq!(
            expected[2].1.primary_key, PENDING_APPROVALS_PRIMARY_KEY,
            "pending-approvals primary key contract should match"
        );
    }

    #[test]
    fn expected_tables_includes_items_when_configured() {
        let tables = TableNames {
            responses: "r".to_owned(),
            conversations: "c".to_owned(),
            items: Some("i".to_owned()),
        };
        let expected = expected_tables(&tables);
        assert_eq!(expected.len(), 4, "should include items and pending-approvals tables");
        assert_eq!(expected[2].0, "i", "third entry should be the items table");
        assert_eq!(
            expected[2].1.primary_key, ITEMS_PRIMARY_KEY,
            "items primary key contract should match"
        );
        assert_eq!(
            expected[3].0, "r_pending_approvals",
            "fourth entry should be the derived pending-approvals table"
        );
    }

    /// Build an [`ActualTable`] from a compact case description.
    ///
    /// `primary_key` pairs a column name with an optional folding reason
    /// (`Some(reason)` marks the column as folding, `None` as safe).
    fn actual_table(
        columns: &[&str],
        primary_key: &[(&str, Option<&str>)],
        immediate: bool,
        unique_indexes: &[(&str, &[&str])],
    ) -> ActualTable {
        ActualTable {
            columns: columns.iter().map(|c| (*c).to_owned()).collect(),
            primary_key: primary_key
                .iter()
                .map(|(name, folding)| ActualKeyColumn {
                    name: (*name).to_owned(),
                    folding: folding.map(str::to_owned),
                })
                .collect(),
            primary_key_immediate: immediate,
            unique_indexes: unique_indexes
                .iter()
                .map(|(name, columns)| ActualUniqueIndex {
                    name: (*name).to_owned(),
                    columns: columns.iter().map(|c| (*c).to_owned()).collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn check_schema_enforces_each_table_dimension() {
        // (case name, actual table, error substrings; empty means it must pass)
        let cases: Vec<(&str, ActualTable, &[&str])> = vec![
            (
                "exact match",
                actual_table(RESPONSES_COLUMNS, &[("id", None)], true, &[]),
                &[],
            ),
            (
                "extra columns tolerated",
                actual_table(
                    &[
                        "tenant_id",
                        "id",
                        "owner_issuer",
                        "owner_subject",
                        "created_at",
                        "model",
                        "response_object",
                        "input",
                        "messages",
                        "extra",
                    ],
                    &[("id", None)],
                    true,
                    &[],
                ),
                &[],
            ),
            (
                "case-insensitive columns and key",
                actual_table(
                    &[
                        "TENANT_ID",
                        "ID",
                        "OWNER_ISSUER",
                        "OWNER_SUBJECT",
                        "CREATED_AT",
                        "MODEL",
                        "RESPONSE_OBJECT",
                        "INPUT",
                        "MESSAGES",
                    ],
                    &[("ID", None)],
                    true,
                    &[],
                ),
                &[],
            ),
            (
                "missing column",
                actual_table(&["tenant_id", "id"], &[("id", None)], true, &[]),
                &["missing columns", "created_at"],
            ),
            (
                "legacy composite primary key",
                actual_table(RESPONSES_COLUMNS, &[("tenant_id", None), ("id", None)], true, &[]),
                &["primary key (tenant_id, id)", "expected (id)"],
            ),
            (
                "wrong primary key",
                actual_table(RESPONSES_COLUMNS, &[("tenant_id", None)], true, &[]),
                &["primary key (tenant_id)", "expected (id)"],
            ),
            (
                "folding key column",
                actual_table(
                    RESPONSES_COLUMNS,
                    &[("id", Some("has type 'citext' (OID 16390)"))],
                    true,
                    &[],
                ),
                &["id", "has type 'citext'"],
            ),
            (
                "deferrable primary key",
                actual_table(RESPONSES_COLUMNS, &[("id", None)], false, &[]),
                &["deferrable primary key"],
            ),
            (
                "extra unique index",
                actual_table(
                    RESPONSES_COLUMNS,
                    &[("id", None)],
                    true,
                    &[("uq_tenant", &["tenant_id"])],
                ),
                &[
                    "unexpected unique index 'uq_tenant'",
                    "only the primary key and the store's own unique indexes may be unique",
                ],
            ),
        ];

        for (name, table, expect_errors) in &cases {
            let input = [("responses", RESPONSES_TABLE, table)];
            let result = check_schema(&input);
            if expect_errors.is_empty() {
                result.unwrap_or_else(|err| panic!("case '{name}' should pass: {err}"));
            } else {
                let msg = result.unwrap_err().to_string();
                assert!(msg.contains("schema validation failed"), "case '{name}': {msg}");
                assert!(msg.contains("database migration required"), "case '{name}': {msg}");
                for needle in *expect_errors {
                    assert!(msg.contains(needle), "case '{name}': expected {needle:?} in {msg}");
                }
            }
        }
    }

    #[test]
    fn check_schema_aggregates_across_tables() {
        let responses = actual_table(RESPONSES_COLUMNS, &[("tenant_id", None), ("id", None)], true, &[]);
        let conversations = actual_table(
            CONVERSATIONS_COLUMNS,
            &[("conversation_id", None)],
            true,
            &[("uq_conv", &["metadata"])],
        );
        let input = [
            ("responses", RESPONSES_TABLE, &responses),
            ("conversations", CONVERSATIONS_TABLE, &conversations),
        ];
        let msg = check_schema(&input).unwrap_err().to_string();
        assert!(msg.contains("responses"), "{msg}");
        assert!(msg.contains("conversations"), "{msg}");
        assert!(msg.contains("uq_conv"), "{msg}");
    }

    #[test]
    fn check_schema_accepts_the_items_position_unique_index() {
        // Column-set matching is order-insensitive, so catalog order does not
        // matter.
        let items = actual_table(
            ITEMS_COLUMNS,
            &[("item_id", None)],
            true,
            &[("idx_i_position", &["position", "conversation_id"])],
        );
        let input = [("i", ITEMS_TABLE, &items)];
        check_schema(&input).expect("the store's own items unique index must be accepted");
    }

    #[test]
    fn check_schema_rejects_an_unexpected_items_unique_index() {
        let items = actual_table(
            ITEMS_COLUMNS,
            &[("item_id", None)],
            true,
            &[("uq_bad", &["tenant_id", "conversation_id", "position"])],
        );
        let input = [("i", ITEMS_TABLE, &items)];
        let msg = check_schema(&input).unwrap_err().to_string();
        assert!(msg.contains("unexpected unique index 'uq_bad'"), "{msg}");
    }

    #[test]
    fn sqlite_type_affinity_classifies_declared_types() {
        // Rule order matters: "INT" wins even inside "POINT", and a type with
        // none of the keywords falls through to NUMERIC.
        let cases = [
            ("INTEGER", SqliteAffinity::Integer),
            ("BIGINT", SqliteAffinity::Integer),
            ("FLOATING POINT", SqliteAffinity::Integer),
            ("TEXT", SqliteAffinity::Text),
            ("VARCHAR(255)", SqliteAffinity::Text),
            ("CLOB", SqliteAffinity::Text),
            ("", SqliteAffinity::Blob),
            ("BLOB", SqliteAffinity::Blob),
            ("REAL", SqliteAffinity::Real),
            ("DOUBLE PRECISION", SqliteAffinity::Real),
            ("FLOAT", SqliteAffinity::Real),
            ("NUMERIC", SqliteAffinity::Numeric),
            ("DECIMAL(10,5)", SqliteAffinity::Numeric),
            ("STRING", SqliteAffinity::Numeric),
        ];
        for (declared, expected) in cases {
            assert_eq!(
                sqlite_type_affinity(declared),
                expected,
                "declared type {declared:?} should resolve to {expected:?}"
            );
        }
    }

    #[test]
    fn sqlite_collation_folds_only_binary_is_safe() {
        assert!(!sqlite_collation_folds("BINARY"), "BINARY preserves distinctness");
        assert!(
            !sqlite_collation_folds("binary"),
            "collation name comparison folds case"
        );
        assert!(sqlite_collation_folds("NOCASE"), "NOCASE folds ASCII case");
        assert!(sqlite_collation_folds("RTRIM"), "RTRIM folds trailing spaces");
        assert!(
            sqlite_collation_folds("custom"),
            "unknown collations are treated as folding"
        );
    }

    #[test]
    fn sqlite_key_column_folding_requires_text_affinity_and_safe_collation() {
        // (declared type, collation, expected folding substring; None means safe)
        let cases: &[(&str, Option<&str>, Option<&str>)] = &[
            ("TEXT", Some("BINARY"), None),
            ("TEXT", None, None),
            ("VARCHAR(255)", Some("BINARY"), None),
            ("TEXT", Some("binary"), None),
            ("INTEGER", Some("BINARY"), Some("INTEGER affinity")),
            ("BLOB", None, Some("BLOB affinity")),
            ("TEXT", Some("NOCASE"), Some("folds distinct values")),
            ("TEXT", Some("RTRIM"), Some("folds distinct values")),
            ("TEXT", Some("custom"), Some("folds distinct values")),
        ];
        for &(declared, collation, expect) in cases {
            let folding = sqlite_key_column_folding(declared, collation);
            match expect {
                None => assert!(
                    folding.is_none(),
                    "type {declared:?} collation {collation:?} should be safe, got {folding:?}"
                ),
                Some(needle) => {
                    let reason =
                        folding.unwrap_or_else(|| panic!("type {declared:?} collation {collation:?} should fold"));
                    assert!(
                        reason.contains(needle),
                        "type {declared:?}: expected {needle:?} in {reason}"
                    );
                },
            }
        }
    }

    #[test]
    fn pg_key_column_folding_allows_only_trusted_text_metadata() {
        // (type OID, type name, collation deterministic, operator class trusted,
        //  expected folding substring; None means safe)
        let cases: &[(i64, &str, Option<bool>, bool, Option<&str>)] = &[
            (25, "text", Some(true), true, None),
            (1043, "varchar", Some(true), true, None),
            (25, "text", None, true, None),
            (16_390, "citext", Some(true), true, Some("citext")),
            (1042, "bpchar", Some(true), true, Some("bpchar")),
            (20_001, "folded_domain", Some(true), true, Some("folded_domain")),
            // The type is checked before the collation, so a disallowed type with
            // a folding collation still reports the type.
            (16_390, "citext", Some(false), false, Some("citext")),
            (25, "text", Some(false), true, Some("non-deterministic collation")),
            (25, "text", Some(true), false, Some("non-default operator class")),
        ];
        for &(type_oid, type_name, deterministic, trusted, expect) in cases {
            let folding = pg_key_column_folding(type_oid, type_name, deterministic, trusted);
            match expect {
                None => assert!(
                    folding.is_none(),
                    "OID {type_oid} ({type_name}) should be safe, got {folding:?}"
                ),
                Some(needle) => {
                    let reason = folding.unwrap_or_else(|| panic!("OID {type_oid} ({type_name}) should fold"));
                    assert!(
                        reason.contains(needle),
                        "OID {type_oid}: expected {needle:?} in {reason}"
                    );
                },
            }
        }
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
        let ddl = generate_ddl(&tables, SqlDialect::Sqlite).expect("valid names should produce DDL");
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
        // Response IDs are globally owner-immutable, so response_id scopes the
        // approval and the owner columns inherit the issuing response owner.
        for expected in [
            "owner_issuer       TEXT NOT NULL",
            "owner_subject      TEXT NOT NULL",
            "response_id        TEXT NOT NULL",
            "approval_id        TEXT NOT NULL",
            "target_fingerprint TEXT NOT NULL",
            "consumed_at        BIGINT",
            "PRIMARY KEY (response_id, approval_id)",
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
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
        let ddl = generate_ddl(&tables, SqlDialect::Sqlite).expect("valid names with items should produce DDL");
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
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
        let err = generate_ddl(&tables, SqlDialect::Sqlite).unwrap_err();
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
