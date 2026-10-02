# Repair duplicate conversation item positions in a legacy database

Before #570, concurrent item appends could assign the same `position` to two
items in one `(tenant_id, conversation_id)` scope. The unique index added by
[#570](https://github.com/praxis-proxy/ai/pull/570) cannot be created while
those rows exist, so store startup fails. This
is an **operator-run data repair**, not an automatic startup migration: changing
persisted ordering without a backup or while writers are active is unsafe.

These statements apply to the **pre-#570, version 1** items table, before its
`(tenant_id, conversation_id, position)` unique index exists. They are for
**historical data recovery only**, not an upgrade to the current schema.
Current releases require a new, empty schema-v5 store; follow the
[recreate-only upgrade policy](schema-migration.md) rather than repairing an
older database for reuse. Do not change the schema-version stamp to make a
newer binary accept an older schema. Index definitions have also changed;
`IF NOT EXISTS` alone will **not** replace an old index definition.

1. Stop every gateway instance writing to this database and take a backup.
2. Substitute the configured items table name for `<items>` throughout the
   chosen backend's SQL. Inspect affected conversations:

   ```sql
   SELECT tenant_id, conversation_id, position, COUNT(*) AS copies
   FROM <items>
   GROUP BY tenant_id, conversation_id, position
   HAVING COUNT(*) > 1;
   ```

3. Run the backend-specific transaction below. It keeps every item and assigns
   positions 1, 2, ... in the existing `(position, item_id)` order **only for
   conversations with duplicates**. That is also the order used to build the
   conversation message cache, so the cache's item order does not change.
   If any statement fails, roll back and investigate instead of starting the
   gateway.

## PostgreSQL

```sql
BEGIN;
LOCK TABLE <items> IN ACCESS EXCLUSIVE MODE;

WITH duplicate_conversations AS (
    SELECT tenant_id, conversation_id
    FROM <items>
    GROUP BY tenant_id, conversation_id
    HAVING COUNT(*) > COUNT(DISTINCT position)
), ranked AS (
    SELECT i.item_id, i.tenant_id, i.conversation_id,
           ROW_NUMBER() OVER (
               PARTITION BY i.tenant_id, i.conversation_id
               ORDER BY i.position, i.item_id
           ) AS next_position
    FROM <items> AS i
    JOIN duplicate_conversations AS d
      ON d.tenant_id = i.tenant_id
     AND d.conversation_id = i.conversation_id
)
UPDATE <items> AS i
SET position = ranked.next_position
FROM ranked
WHERE i.item_id = ranked.item_id
  AND i.tenant_id = ranked.tenant_id
  AND i.conversation_id = ranked.conversation_id
  AND i.position IS DISTINCT FROM ranked.next_position;

CREATE UNIQUE INDEX IF NOT EXISTS idx_<items>_position
    ON <items>(tenant_id, conversation_id, position);
COMMIT;
```

## SQLite

Requires SQLite 3.35 or later. `MATERIALIZED` fixes the ranking snapshot
before the update changes any positions.

```sql
BEGIN IMMEDIATE;

WITH duplicate_conversations AS (
    SELECT tenant_id, conversation_id
    FROM <items>
    GROUP BY tenant_id, conversation_id
    HAVING COUNT(*) > COUNT(DISTINCT position)
), ranked AS MATERIALIZED (
    SELECT i.item_id, i.tenant_id, i.conversation_id,
           ROW_NUMBER() OVER (
               PARTITION BY i.tenant_id, i.conversation_id
               ORDER BY i.position, i.item_id
           ) AS next_position
    FROM <items> AS i
    JOIN duplicate_conversations AS d
      ON d.tenant_id = i.tenant_id
     AND d.conversation_id = i.conversation_id
)
UPDATE <items> AS i
SET position = (
    SELECT r.next_position FROM ranked AS r
    WHERE r.item_id = i.item_id
      AND r.tenant_id = i.tenant_id
      AND r.conversation_id = i.conversation_id
)
WHERE EXISTS (
    SELECT 1 FROM ranked AS r
    WHERE r.item_id = i.item_id
      AND r.tenant_id = i.tenant_id
      AND r.conversation_id = i.conversation_id
      AND r.next_position <> i.position
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_<items>_position
    ON <items>(tenant_id, conversation_id, position);
COMMIT;
```

Run the inspection query again. It must return no rows. Check that the item
count still matches the backup before using the recovered historical data.
Do not reuse this repaired database with a current release; provision a new
store as described in [schema migration](schema-migration.md).
