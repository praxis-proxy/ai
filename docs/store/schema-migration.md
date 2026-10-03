# Responses Store Schema Upgrade

Current releases require schema version 5 and refuse to start against any older
store. Schema upgrades are recreate-only: Praxis does not migrate an existing
store in place, and it fails closed rather than auto-stamping a newer version.

Schema version 5 adds the owner-scoped SSE replay event-log table used by
[streaming replay](../architecture/response-store.md#sse-replay) on top of the
version-4 layout. Version 4 (shipped after Praxis AI v0.4.1's version 3) changed
the conversation-item layout and cannot reuse the older tables.

Existing response and conversation state is intentionally disposable. To
upgrade:

1. Stop every proxy instance that uses the store.
2. Back up the old database if its contents may still be needed for audit or
   manual recovery.
3. Provision an empty, dedicated PostgreSQL database or a new SQLite database
   file. If the existing database is dedicated to Praxis AI, it may instead be
   dropped and recreated after the backup is verified.
4. Update `database_url` if the replacement uses a new location.
5. Start the proxy. Startup provisioning creates and validates the complete
   schema-v5 table set (including the replay event-log table) before the service
   becomes ready.

Do not update only the schema-version row, and do not reuse v0.4.1 tables
unchanged. Their v3 layout lacks the owner-scoped conversation-item identity
and uniqueness constraints required by v4.

The same recreate-only policy applies to standalone `openai_response_store`,
standalone `openai_conversations`, and compatible deployments where the two
filters share one backend. When a PostgreSQL database contains unrelated data,
create a new database for Praxis AI rather than dropping the shared database.
