# Response Store

Durable persistence for OpenAI Responses API responses,
enabling retrieval (`GET`), deletion (`DELETE`), and
input-item pagination across proxy restarts.

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

The production binary scans every listener's reachable filter chains, validates
the selected backend, and provisions pools on the serving runtime before the
store readiness gate admits traffic. Request-path filters resolve an
owner-scoped handle from their listener registry and never construct SQL pools.
Responses and Conversations can share one pool when their effective backend
configuration matches. The provisioner promotes the two filter-shaped configs
to one combined table set: Responses supplies the responses table,
Conversations supplies the items table, and both must select the same
conversations table. Both registry names then resolve the same backend lease and
SQL pool.

## Backend Features

The default `praxis-ai-proxy` build uses `full`, so PostgreSQL-backed Responses
and Conversations are available in the production binary while SQLite remains
opt-in. Explicit `--no-default-features` builds provide the four supported
persistence profiles:

| Profile | Feature selection | SQL backends |
|---------|-------------------|--------------|
| Backend-free | `standard,openai-all` | None; contracts, record helpers, and filters only |
| PostgreSQL-only | `standard,openai-all,store-postgres` | PostgreSQL through SQLx native TLS |
| SQLite-only | `standard,openai-all,store-sqlite` | SQLite |
| Combined | `standard,openai-all,store-all` | PostgreSQL and SQLite |

The released container image builds `full,store-sqlite`, so it carries both
backends. SQLite configurations use a database path under the image's writable
state directory, as described below.

The backend-free profile is an internal composition and testing lane. A config
that selects an implementation absent from the binary is rejected during
pipeline construction with an actionable backend-unavailable diagnostic.

SQLite examples require an explicit build from source:

```console
cargo run -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite -- \
  -c examples/configs/openai/responses/response-store.yaml
```

Store configuration reload is generation-based. The serving runtime provisions
and validates the replacement generation first, the watcher atomically swaps
pipelines only after it succeeds, and the old generation retains its leases
until request-held pipeline references drain. Identical configurations reuse
the cached pool; changed configurations retire the previous pool after drain.
Expanding an active in-memory SQLite Responses store into a combined Responses
and Conversations topology requires a restart. A replacement pool would be a
different transient database and would otherwise lose the active store state.

The examples use a relative `database_url` such as
`sqlite://responses.db?mode=rwc`, which resolves against the working directory.
In the container image the working directory is the root-owned `/etc/praxis`,
so point `database_url` at the writable state directory instead:
`sqlite:///var/lib/praxis/responses.db?mode=rwc`. Mount a volume there to
keep the database across container restarts.

A configuration that selects a backend absent from the binary is rejected
while the filter pipeline is constructed, before the proxy serves traffic.

## Request Phases

The filter spans three Pingora phases, each refining
the persistence decision as new information arrives:

### `on_request`

- Reads classifier metadata to determine if the
  request is persistable (POST, responses format,
  store enabled).
- Handles `GET /v1/responses/{id}` retrieval and
  `GET /v1/responses/{id}/input_items` pagination
  directly from the store. Query parameters on
  `GET /v1/responses/{id}` are validated before
  retrieval: `stream=false` and empty queries return
  the plain JSON record; `stream=true` replays the
  stored SSE event log (see [SSE Replay](#sse-replay));
  `starting_after` is accepted only alongside
  `stream=true` as a replay cursor and is otherwise
  rejected with a 400. `include`, `include_obfuscation`,
  and unknown parameters are rejected with a 400.
- Handles `DELETE /v1/responses/{id}` locally.
- Resolves the owner-scoped store service from the listener registry.
- Rejects fail-closed if a request requiring persistence reaches an
  unprovisioned registry. The readiness gate normally prevents that state from
  receiving traffic.

### `on_response`

- Re-checks skip conditions with response headers.
- Non-2xx responses, or responses that are neither JSON
  nor SSE, set `responses.skip_persist` and bail early.

### `on_response_body`

- For a non-streaming response, extracts the record from
  buffered JSON at end-of-stream.
- For a `stream: true` response, captures each
  normalized SSE event as it passes the store seam
  (already sequence-stamped by `openai_stream_events`
  upstream on the response path). At the terminal event,
  builds the record from accumulated response state and
  flushes the replay log after persisting the record,
  before releasing the terminal event.
- Persists synchronously via `block_in_place`
  before returning to Pingora.
- Non-persistable exchanges release chunks via
  `FilterAction::Release` to avoid holding
  pass-through traffic.

## SSE Replay

A response created with `stream: true` and stored by the proxy retains its
normalized SSE event log. After the original response finishes and the log is
persisted, `GET /v1/responses/{id}?stream=true` replays the stored events in
sequence order. This GET replays a recording; it does not start or continue
model generation. A plain GET without `stream=true` returns the stored response
as JSON. `starting_after=N` returns only events whose `sequence_number > N`.
If `N` precedes the terminal event, replay includes that event; at or beyond
the terminal sequence number, it returns an empty SSE body. Replay does not
reconstruct deltas. The body pages the store, so serving a replay does not
load the whole log at once.

The original foreground POST keeps the captured events in request scope and
persists the response record and event log at its terminal event. While the
response is still generating, there is normally no stored record, so a GET
returns 404 even if the client already received the response ID. If the record
exists but has no complete replay log, `?stream=true` returns 400. A dropped
foreground connection may interrupt the original request; this feature does
not keep generation running or guarantee that a replayable log will be saved.
Reconnecting to a stream while generation continues requires background
execution, which is not implemented here.

Capture happens at the store filter's response-body seam, which sits after
`openai_stream_events` on the response path, so the persisted events match the
frames the client observed live (including any incremental rewrites such as
`previous_response_id` restore). The log is owner-scoped like every other
record: a mismatched owner receives a 404. Deleting the response removes its
event log. A response with no terminal event never appears as complete.

The log is bounded by two filter options, `max_event_count` (default 10,000)
and `max_event_bytes` (default 16 MiB). A stream that exceeds either bound stops
event capture, so its terminal event is never recorded and the response becomes
non-replayable. Retrieving a non-replayable response with `?stream=true` —
because it exceeded a bound, or was created without `stream: true`, or otherwise
has no terminal event — returns a 400 `invalid_request_error` rather than a
partial or fabricated stream. The live client stream and the plain JSON record
are unaffected by the bounds.

The replay log lives in a dedicated owner-scoped table created by the startup
DDL (schema version 5). The [`ResponseStore`] trait gains `append_events`,
`list_events_after`, and `event_log_status` for the capture, paged serving, and
replayability gate respectively.

[`ResponseStore`]: ../../store/src/traits.rs

## Threading Model

The response body hook (`on_response_body`) is a
synchronous `fn`, not `async fn`. The filter bridges
to the async store trait using:

```rust
let handle = tokio::runtime::Handle::current();
tokio::task::block_in_place(|| {
    handle.block_on(store.upsert_response(record))
})
```

This guarantees the record is durable before the
client observes the completed response, preventing
races where a subsequent `DELETE` arrives before the
upsert completes.

## Store Provisioning

Pools are created eagerly by the server-owned provisioner, never by a request
filter. Each backend factory validates its typed configuration before pool
creation. The lifecycle cache retries `BackendError::Transient` within a
bounded attempt budget; exhaustion becomes terminal `Unavailable`. Invalid
configuration, unsupported schemas, and other permanent initialization
failures are terminal immediately and are not retried forever. Concurrent cold
misses for the same effective backend key share one singleflight build, so a
generation never opens duplicate pools for matching listeners. Aggregate store
readiness becomes ready only after every configured listener owns a live
generation lease. A terminal initial-generation failure rejects process
startup with the backend diagnostic instead of leaving listeners alive behind
persistent 503 responses. A later reload provisioning failure rejects only the
candidate generation; the active generation keeps serving and a subsequent
reload can retry. Pool opening and schema preparation have a 30-second
end-to-end deadline so a database lock cannot block reload or shutdown
indefinitely.

## Storage Backends

### SQLite

File-backed or in-memory. In-memory databases use a
single-connection pool to avoid cross-connection
isolation. Response payload columns use `BLOB`, allowing the configured
compression layer to store encoded bytes without text conversion.

### PostgreSQL

Connection-pooled via `sqlx::PgPool`. Response payload columns use `BYTEA`.
Upsert uses `ON CONFLICT (id) DO UPDATE` and updates only when the stored and
incoming owner triples match, preserving globally unique response IDs without
allowing ownership changes. Supports configurable
`SslMode` (`disable`, `prefer`, `require`,
`verify-ca`, `verify-full`), custom root CA
certificates, and client-certificate (mutual TLS)
authentication. See the [PostgreSQL cryptographic
boundary](postgres-cryptographic-boundary.md) for the
certificate-authentication compliance profile that
keeps password cryptography off the connection path.

SSRF protections reject DNS hostnames, localhost,
loopback, private, link-local, and unspecified
addresses by default. `allow_private_database_url`
opts in for development, but cloud metadata,
unspecified, and multicast addresses remain blocked.
Host validation is re-run
on every connection attempt to guard against DNS
rebinding.

## Ownership Isolation

Every read, write, and delete is scoped by the complete trusted owner identity:
`tenant_id`, `owner_issuer`, and `owner_subject`. Responses and conversations
use globally unique resource IDs; owner-preserving conflict predicates reject
an attempt to reuse an existing ID under another owner. Conversation items use
an owner-qualified composite key. Lookups return `None` for a mismatched owner
to prevent information leakage. Single-tenant deployments use a `"default"`
tenant sentinel while retaining issuer and subject isolation.

## Body Buffering

The filter declares `BodyMode::StreamBuffer` with a
64 MiB ceiling globally. Non-streaming Responses API
payloads are bounded by output token limits (typically
under 2 MiB). Non-persistable exchanges release chunks
immediately via `FilterAction::Release` so
pass-through traffic is not held.

## Key Files

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
