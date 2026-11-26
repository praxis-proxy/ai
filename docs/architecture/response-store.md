# Response store

Durable persistence for OpenAI Responses API responses, so retrieval (`GET`),
deletion (`DELETE`), and input-item pagination survive proxy restarts.

## Design

Persistence is split into explicit dependency layers:

```text
Responses / Conversations filters
  |
  +-- owner-scoped store handle and record-assembly helpers
  |
praxis-ai-store
  |-- SQL-free records, traits, registries, and backend factory contracts
  v
praxis-ai-store-lifecycle
  |-- generation leases, effective-config deduplication, bounded retry
  v
praxis-ai-store-backends
  |
  +-- PostgreSQL and SQLite pools, schemas, and TLS
```

The production binary scans every listener's reachable filter chains,
validates the selected backend, and provisions pools on the serving runtime
before the store readiness gate admits traffic. Request-path filters resolve an
owner-scoped handle from their listener registry and never construct SQL
pools.

Responses and Conversations share one pool when their effective backend
configuration matches. The provisioner then promotes the two filter configs to
one combined table set: Responses supplies the responses table, Conversations
supplies the items table, and both must select the same conversations table.
Both registry names resolve the same backend lease and SQL pool.

`praxis_ai_apis::store::plan` exposes offline discovery, config normalization,
compatible-store sharing, and SQL namespace validation, so the proxy server and
external serving transports reuse these rules without copying schema
knowledge. Each transport still owns its registry, readiness, serving runtime,
and lifecycle leases; reload-specific active-key preservation stays in the
proxy server. Discovery keeps caller policy: the proxy server skips malformed
nested entries and unparseable store references and keeps its cross-store
collision checks, while strict transports can reject them and use the stricter
single-store validator.

## Backend features

Responses, Conversations, and their stores are experimental and absent from the
default `praxis-ai-proxy` build. Four feature selections give the supported
persistence profiles:

| Profile | Feature selection | SQL backends |
|---------|-------------------|--------------|
| Backend-free | `standard,openai-all` | None; contracts, record helpers, and filters only |
| PostgreSQL-only | `standard,openai-all,store-postgres` | PostgreSQL through SQLx native TLS |
| SQLite-only | `standard,openai-all,store-sqlite` | SQLite |
| Combined | `standard,openai-all,store-all` | PostgreSQL and SQLite |

The backend-free profile is an internal composition and testing lane. A config
that selects a backend absent from the binary is rejected with an actionable
backend-unavailable diagnostic while the pipeline is constructed, before the
proxy serves traffic.

The standard container image builds `full` without Responses or stores. The
`-experimental` image tags build `full,openai-all,store-all`; pass
`--build-arg PRAXIS_AI_FEATURES=full,openai-all,store-all` to do the same
locally. SQLite examples need a build from source:

```console
cargo run -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite -- \
  -c examples/configs/openai/responses/response-store.yaml
```

The examples use a relative `database_url` such as
`sqlite://responses.db?mode=rwc`, which resolves against the working directory.
In the container image that is the root-owned `/etc/praxis`, so point
`database_url` at the writable state directory instead:
`sqlite:///var/lib/praxis/responses.db?mode=rwc`. Mount a volume there to keep
the database across container restarts.

Store configuration reload is generation-based. The serving runtime provisions
and validates the replacement generation first, the watcher swaps pipelines
atomically only after that succeeds, and the old generation keeps its leases
until request-held pipeline references drain. Identical configurations reuse
the cached pool; changed configurations retire the previous pool after drain.
Expanding an active in-memory SQLite Responses store into a combined Responses
and Conversations topology requires a restart, because a replacement pool would
be a different transient database and the active store state would be lost.

## Store provisioning

The server-owned provisioner creates pools eagerly; a request filter never
does. Each backend factory validates its typed configuration before pool
creation. The lifecycle cache retries `BackendError::Transient` within a
bounded attempt budget, and exhaustion becomes terminal `Unavailable`. Invalid
configuration, unsupported schemas, and other permanent initialization
failures are terminal immediately. Concurrent cold misses for the same
effective backend key share one singleflight build, so a generation never opens
duplicate pools for matching listeners.

Aggregate store readiness turns ready only after every configured listener owns
a live generation lease. A terminal failure in the initial generation stops
process startup with the backend diagnostic, so listeners are not left serving
persistent 503 responses. A failure while provisioning a later reload rejects
only the candidate generation; the active generation keeps serving and a later
reload can retry. Pool opening and schema preparation have a 30-second
end-to-end deadline so a database lock cannot block reload or shutdown
indefinitely.

## Request phases

The filter spans three Pingora phases, each refining the persistence decision:

### `on_request`

- Reads classifier metadata to decide whether the request is persistable
  (POST, responses format, store enabled).
- Serves `GET /v1/responses/{id}` retrieval and
  `GET /v1/responses/{id}/input_items` pagination directly from the store.
  Query parameters on `GET /v1/responses/{id}` are validated first:
  `stream=false` and empty queries return the plain JSON record; `stream=true`
  replays the stored SSE event log (see [SSE replay](#sse-replay));
  `starting_after` is accepted only with `stream=true` as a replay cursor and
  otherwise gets a 400. `include`, `include_obfuscation`, and unknown
  parameters get a 400.
- Handles `DELETE /v1/responses/{id}` locally.
- Resolves the owner-scoped store service from the listener registry.
- Fails closed if a request that needs persistence reaches an unprovisioned
  registry. The readiness gate normally keeps that state from receiving
  traffic.

### `on_response`

- Re-checks skip conditions against response headers.
- Non-2xx responses, or responses that are neither JSON nor SSE, set
  `responses.skip_persist` and stop early.

### `on_response_body`

- For a non-streaming response, extracts the record from buffered JSON at
  end-of-stream.
- For a `stream: true` response, captures each normalized SSE event as it
  passes the store seam (already sequence-stamped by `openai_stream_events`
  upstream on the response path). At the terminal event, it builds the record
  from accumulated response state, persists it, then flushes the replay log
  before releasing the terminal event.
- Persists synchronously via `block_in_place` before returning to Pingora (see
  [Threading model](#threading-model)).
- Non-persistable exchanges release chunks via `FilterAction::Release` so
  pass-through traffic is not held.

## SSE replay

A response created with `stream: true` and stored by the proxy keeps its
normalized SSE event log. Once the original response finishes and the log is
persisted, `GET /v1/responses/{id}?stream=true` replays the stored events in
sequence order. It replays a recording and never starts or continues model
generation. `starting_after=N` returns only events whose
`sequence_number > N`. If `N` precedes the terminal event, replay includes that
event; at or beyond the terminal sequence number, it returns an empty SSE body.
Replay does not reconstruct deltas. The body pages the store, so a replay never
loads the whole log at once.

The original foreground POST keeps captured events in request scope and
persists the response record and event log at its terminal event. While the
response is still generating there is normally no stored record, so a GET
returns 404 even if the client already has the response ID. A dropped
foreground connection may interrupt the original request; nothing keeps
generation running or guarantees a replayable log. Reconnecting to a stream
while generation continues requires background execution, which is not
implemented.

Capture happens at the store filter's response-body seam, after
`openai_stream_events` on the response path, so the persisted events match the
frames the client saw live (including incremental rewrites such as
`previous_response_id` restore). The log is owner-scoped like every other
record: a mismatched owner gets a 404. Deleting the response removes its event
log.

Two filter options bound the log: `max_event_count` (default 10,000) and
`max_event_bytes` (default 16 MiB). A stream that exceeds either stops event
capture, so its terminal event is never recorded and the response becomes
non-replayable. A response with no terminal event (over a bound, created
without `stream: true`, or otherwise incomplete) never appears as complete, and
`?stream=true` returns a 400 `invalid_request_error` instead of a partial or
fabricated stream. The bounds do not affect the live client stream or the plain
JSON record.

The replay log lives in a dedicated owner-scoped table created by the startup
DDL (schema version 5). The [`ResponseStore`] trait has `append_events`,
`list_events_after`, and `event_log_status` for capture, paged serving, and the
replayability check.

[`ResponseStore`]: ../../store/src/traits.rs

## Threading model

The response body hook (`on_response_body`) is a synchronous `fn`, not
`async fn`. The filter bridges to the async store trait with:

```rust
let handle = tokio::runtime::Handle::current();
tokio::task::block_in_place(|| {
    handle.block_on(store.upsert_response(record))
})
```

The record is durable before the client sees the completed response, so a
following `DELETE` cannot arrive before the upsert completes.

## Storage backends

### SQLite

File-backed or in-memory. In-memory databases use a single-connection pool to
avoid cross-connection isolation. Response payload columns use `BLOB`, so the
configured compression layer can store encoded bytes without text conversion.

File-backed databases enable WAL so a snapshot reader does not block a
writer's commit; in-memory databases keep their native journal. File-backed
databases must use local storage that supports WAL shared memory, not a network
filesystem. Back up with SQLite's backup facilities: copying an active
database file alone can miss committed changes still in its WAL. Long-lived
readers can delay checkpoints and grow the WAL, so monitor disk usage and
transaction duration. WAL does not remove contention between writers or
promise a latency bound.

### PostgreSQL

Connection-pooled via `sqlx::PgPool`. Response payload columns use `BYTEA`.
Upsert uses `ON CONFLICT (id) DO UPDATE` and updates only when the stored and
incoming owner triples match, which keeps response IDs globally unique without
allowing ownership changes. Supports configurable `SslMode` (`disable`,
`prefer`, `require`, `verify-ca`, `verify-full`), custom root CA certificates,
and client-certificate (mutual TLS) authentication. See the
[PostgreSQL cryptographic boundary](postgres-cryptographic-boundary.md) for the
certificate-authentication compliance profile that keeps password cryptography
off the connection path.

By default the SSRF checks reject DNS hostnames, localhost, loopback, private,
link-local, and unspecified addresses. `allow_private_database_url` opts in for
development, but cloud metadata, unspecified, and multicast addresses stay
blocked. Host validation re-runs on every connection attempt to guard against
DNS rebinding.

## Conversation history

Conversation item rows are authoritative for Responses continuation. History is
read in `(position, item_id)` order from one owner-scoped snapshot: a read
transaction on SQLite, a repeatable-read, read-only transaction on PostgreSQL.
SQLx streams rows into the owned history vector instead of collecting all raw
SQL rows alongside their decoded JSON.

Replay and persistence share unchanged JSON through item-level copy-on-write
history. Legacy normalization and content/tool rewrites detach only changed
items, and hosted-tool items stay in exact persisted order. Native request
serialization borrows the projected history directly; independent JSON values
are materialized only at the durable-record ownership boundary. Rehydration
does not serialize history a second time for sizing and imposes no total
item-count or byte-size limit on history. The backend handles model context
overflow through the client's `truncation` setting, which is preserved for
native Responses requests. Other gateway limits still apply: file-search
continuations enforce `max_state_bytes` over replay and persisted state, and
transport, request-body, and storage safeguards remain.

Adding or deleting items assigns positions and invalidates the retired
`messages` cache in the same transaction, without decoding other item payloads.
Clearing the cache after the last deletion keeps stale items from reappearing.
An already empty cache or preserved legacy prefix is left untouched, so
ordinary mutations do not rewrite the conversation row. Per-request item limits
and exact tenant/issuer/subject isolation still apply.

### Legacy cache records

The schema and its version are unchanged. A conversation with both item rows
and a cache replays from item rows. Cache-only legacy records stay readable,
including Responses-only configurations without an items table. The first
append to a nonempty cache-only record preserves its original array as an
internal `legacy_messages` prefix in the existing JSON column. That one-time
mutation reads and rewrites the legacy cache, not item rows; later mutations
keep the prefix without decoding it. History reads prepend the prefix to
ordered item rows, including after the last new item is deleted. No startup
migration, schema change, or item-row backfill runs automatically.

The `conversation_history` store method is the replay boundary. Raw
`get_conversation` still returns metadata and legacy cache data, not
reconstructed item history.

Startup validation checks the conversation `messages` column as well as table
structure. PostgreSQL requires built-in `TEXT` (not JSON/JSONB or a domain).
SQLite requires TEXT affinity, accepting VARCHAR/CLOB and rejecting INTEGER,
BLOB, and JSONB declarations. An incompatible custom schema fails before the
schema version is stamped and needs an explicit, reviewed schema repair. No
type conversion or history rewrite runs automatically.

### Upgrade and rollback

A mixed-version rolling upgrade against the same conversation tables is not
supported. An older writer can rebuild `messages` from item rows and
permanently erase the preserved legacy prefix, and an older reader cannot
replay the new representation. Before upgrading, stop conversation traffic,
drain in-flight requests and all old writers, and back up the tables. Replace
every instance sharing those tables before resuming traffic. If old and new
binaries must run at the same time, give them separate tables.

Before rolling back to a binary that reads only the cache, stop conversation
traffic and drain all new writers. Rebuild each affected cache from its legacy
prefix and ordered item rows, using a backup and the deployment's migration
procedure, then replace every instance before resuming traffic. An old binary
must not serve continuations against invalidated caches.

### Invalid legacy cache recovery

A cache-only row containing unsupported JSON objects or scalar values rejects
appends with a serialization error. The failed transaction keeps both the
original cache and item positions and never replaces history with an empty
array. Item-backed conversations keep ignoring a stale array cache.

To recover, stop traffic for the affected conversation, back up its row and
items, and confirm the intended complete legacy history from trustworthy
evidence. For a cache-only row, call the store's owner-scoped
`compare_and_swap_conversation_messages` with the exact observed invalid value
and the verified replacement array. A stale expected value or a different owner
returns false; reread and investigate instead of forcing the write. If history
cannot be recovered, leave the conversation rejected rather than lose data
silently. This repair path must not overwrite the protected `legacy_messages`
prefix. Confirm complete history and a successful append before resuming
traffic. This is an integration/operator procedure; there is no public HTTP
repair endpoint.

## Ownership isolation

Every read, write, and delete is scoped by the complete trusted owner identity:
`tenant_id`, `owner_issuer`, and `owner_subject`. Responses and conversations
use globally unique resource IDs, and owner-preserving conflict predicates
reject reuse of an existing ID under another owner. Conversation items use an
owner-qualified composite key. Lookups return `None` for a mismatched owner so
nothing leaks. Single-tenant deployments use a `"default"` tenant sentinel and
keep issuer and subject isolation.

## Body buffering

The filter declares `BodyMode::StreamBuffer` with a 64 MiB ceiling globally. On
a listener with an agentic
[retained-payload budget](responses-retained-payload.md), the buffered response
cap can be lower. Output token limits bound non-streaming Responses API
payloads (typically under 2 MiB).

## TTL and automatic purging

Store filters accept optional `ttl`, `sweep_interval`, and `sweep_batch_size`
fields that turn on automatic purging:

- `ttl: <duration>`: retention period (humantime format: `7d`, `24h`, `30m`,
  etc.). Omit it to disable the reaper.
- `sweep_interval: <duration>`: how often the reaper runs (default: `1h`).
- `sweep_batch_size: <count>`: maximum records purged per sweep batch
  (default: `1000`).

TTL is measured from record creation time, not last access. Activity does
not extend retention, so a conversation or response older than `ttl` is purged
even if it was used recently. TTL is not a storage-capacity guarantee; tenant
byte quotas and capacity accounting are tracked in #1626.

One reaper coordinator per server process runs one reaper task per distinct
TTL-enabled store backend, keyed on the factory's `EffectiveConfigKey`. Two
filters on one physical database with different table sets (e.g. a different
`items_table` or `pool`) therefore each get their own reaper, and every
backend is swept exactly once. Each reaper sweeps all record types
(responses, conversations, conversation items, events, approvals) from its
backend's tables, deleting records whose `created_at` is older than
`now - ttl` in bounded batches. The coordinator persists its policy (ttl,
sweep_interval, sweep_batch_size) on each config commit; policy changes take
effect when the pipeline reloads. Reapers stop gracefully on shutdown and
across config reloads.

The reaper sweeps each table independently and does not cascade. Each
table is swept by its own `created_at`, in its own bounded, per-batch-committed
transaction, with no foreign-key cascade. The reaper repeats batches per table
until that table has no rows older than the cutoff or the batch size is
reached. A child row's `created_at` (a conversation item, a response event, a
pending approval) is caller-supplied and can be well after its parent's, so a
parent such as a response can be swept while a child still inside the TTL
window survives. That orphaned child (its `response_id`/parent id points at a
deleted row) persists until its own `created_at` passes the cutoff, which can be
much longer than one sweep interval. Creation-time semantics accept this
because orphaned children of a dead parent id are inert.

The request-path delete endpoints differ from the reaper and from each other.
`delete_response` cascades: it removes the response row with its dependent
pending-approval and event rows in one operation. `delete_conversation` does
not cascade: it deletes only the conversation row and keeps the item rows (see
`ConversationItemStore::delete_conversation`).

Distinct backends can still share a physical table: two stores with different
`responses_table` but the same `conversations_table` resolve to different keys
and both sweep that shared table. So that a shorter TTL cannot delete rows that
a longer retention (or a store with no `ttl`) still needs, provisioning
validates physical-table ownership. Within one database, no two stores may
assign different policies to the same physical table, including TTL-enabled
versus TTL-disabled; such a configuration is rejected before serving starts.
The same policy on a shared table is allowed. For independent retention
policies on one database, give each store a fully disjoint table set or its own
database.

Future quota accounting (#1626) must credit storage released by TTL purges in
the same atomic commit that removes the rows, so accounting stays exactly-once
even if a purge is interrupted. TTL and capacity limits are independent;
operators may configure both a time-based TTL and a future tenant-scoped
capacity quota.

## Key files

- `apis/src/openai/responses/store/filter.rs`: HTTP filter lifecycle
- `apis/src/service/responses/`: Responses record assembly and input-item listing
- `apis/src/service/conversations/`: conversation item-record assembly and validation
- `store/`: SQL-free owner-scoped handle, records, registries, and factory traits
- `store-lifecycle/`: cache, retry, generation leases, reuse, and retirement
- `store-backends/`: SQLx implementations, pools, schemas, TLS, and factories
- `server/src/store_provision.rs`: listener planning, readiness, and serving-runtime provisioning
- `server/src/reload.rs`: atomic pipeline-generation reload orchestration

## Related

- [AI Inference](ai-inference.md)
- [PostgreSQL cryptographic boundary](postgres-cryptographic-boundary.md)
- [Features](../features.md)
