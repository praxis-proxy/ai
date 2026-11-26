# Responses Store Schema Upgrade

Current releases require schema version 5 and refuse to start against an older
store. Upgrades are recreate-only: Praxis does not migrate a store in place,
and it fails closed rather than stamping a newer version on an old schema.

Version 5 adds the owner-scoped SSE replay event-log table used by
[streaming replay](../../architecture/response-store.md#sse-replay) to the
version 4 layout. Version 4, which followed Praxis AI v0.4.1's version 3,
changed the conversation-item layout, so older tables cannot be reused.

Response and conversation state is disposable by design. To upgrade:

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

Do not edit only the schema-version row to make a newer binary accept an older
schema, and do not reuse v0.4.1 tables. Their v3 layout lacks the owner-scoped
conversation-item identity and uniqueness constraints that v4 requires.

The policy is the same for standalone `openai_responses_store`, standalone
`openai_conversations`, and deployments where the two filters share one
backend. If a PostgreSQL database also holds unrelated data, create a new
database for Praxis AI instead of dropping the shared one.

To recover historical pre-#570 data, see
[repair duplicate item positions](legacy-item-position-repair.md). That repair
is not an upgrade path to schema v5.
