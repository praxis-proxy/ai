# Responses Store Schema Migration (v2 → v3)

Schema **version 3** stores the responses store's JSON payload columns as
native binary columns instead of `TEXT`:

| Table                | Columns changed to binary                 |
| -------------------- | ----------------------------------------- |
| responses            | `response_object`, `input`, `messages`    |

Binary columns let the optional zstd compression store a raw zstd frame
(and uncompressed rows store raw JSON bytes). The
column type is `BLOB` on SQLite and `BYTEA` on PostgreSQL.

The proxy **refuses to start** against a database still stamped at
version 2 — the store initialization fails with:

```
schema version mismatch in '<responses_table>_schema_version': stored
version 2, expected 3; database migration required
```

Migration is a one-time, operator-run step. It is not applied
automatically on startup. Take a backup before running it.

Substitute your configured responses table name (`openai_responses` by
default) for the `<responses_table>` placeholder below.

## PostgreSQL

PostgreSQL is strictly typed, so the columns must be altered from `TEXT`
to `BYTEA`. `convert_to(col, 'UTF8')` reinterprets the existing JSON text
as its UTF-8 bytes. Run the following in one transaction:

```sql
BEGIN;

ALTER TABLE <responses_table>
    ALTER COLUMN response_object TYPE BYTEA USING convert_to(response_object, 'UTF8'),
    ALTER COLUMN input           TYPE BYTEA USING convert_to(input, 'UTF8'),
    ALTER COLUMN messages        TYPE BYTEA USING convert_to(messages, 'UTF8');

UPDATE <responses_table>_schema_version SET version = 3;

COMMIT;
```

## SQLite

SQLite uses dynamic typing, so a value keeps the storage class it was
written with regardless of the column's declared affinity. `CAST(col AS
BLOB)` rewrites each stored `TEXT` value to its `BLOB` storage class.
Run the following in one transaction:

```sql
BEGIN;

UPDATE <responses_table>
SET response_object = CAST(response_object AS BLOB),
    input           = CAST(input AS BLOB),
    messages        = CAST(messages AS BLOB);

UPDATE <responses_table>_schema_version SET version = 3;

COMMIT;
```

## The "openai_conversations" Filter

The `openai_conversations` filter does not store responses, but it shares
the response-store schema and therefore generates an internal,
always-empty `<conversations_table>_unused_responses` table (default
`openai_conversations_unused_responses`) with its own
`<conversations_table>_unused_responses_schema_version`. That version is
gated by the same global schema version, so an existing conversations
deployment stamped at version 2 also refuses to start until its generated
table is stamped at version 3.

Because the generated table is never written, no data conversion is
needed — only the schema version has to be bumped.

```sql
UPDATE openai_conversations_unused_responses_schema_version SET version = 3;
```

The conversations (`<conversations_table>`) and items tables are
unchanged by this migration and must not be altered.
