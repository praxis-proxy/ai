# Response Store

Durable persistence for OpenAI Responses API responses,
enabling retrieval (`GET`), deletion (`DELETE`), and
input-item pagination across proxy restarts.

## Design

Persistence is split into explicit dependency layers:

```text
Responses / Conversations filters
  |
  +-- owner-scoped service APIs
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
| Backend-free | `standard,openai-all` | None; contracts, services, and filters only |
| PostgreSQL-only | `standard,openai-all,store-postgres` | PostgreSQL through SQLx native TLS |
| SQLite-only | `standard,openai-all,store-sqlite` | SQLite |
| Combined | `standard,openai-all,store-all` | PostgreSQL and SQLite |

The backend-free profile is an internal composition and testing lane. A config
that selects an implementation absent from the binary is rejected during
pipeline construction with an actionable backend-unavailable diagnostic.

SQLite examples require an explicit build:

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

## Request Phases

The filter spans three Pingora phases, each refining
the persistence decision as new information arrives:

### `on_request`

- Reads classifier metadata to determine if the
  request is persistable (POST, responses format,
  store enabled, non-streaming).
- Handles `GET /v1/responses/{id}` retrieval and
  `GET /v1/responses/{id}/input_items` pagination
  directly from the store. Query parameters on
  `GET /v1/responses/{id}` are validated before
  retrieval: `stream=false` and empty queries are
  accepted, while `stream=true`, `include`,
  `starting_after`, `include_obfuscation`, and
  unknown parameters are rejected with a 400
  response.
- Handles `DELETE /v1/responses/{id}` locally.
- Resolves the owner-scoped store service from the listener registry.
- Rejects fail-closed if a request requiring persistence reaches an
  unprovisioned registry. The readiness gate normally prevents that state from
  receiving traffic.

### `on_response`

- Re-checks skip conditions with response headers.
- Non-2xx or non-JSON responses set
  `responses.skip_persist` and bail early.

### `on_response_body`

- At end-of-stream, extracts the record from the
  buffered response JSON.
- Persists synchronously via `block_in_place`
  before returning to Pingora.
- Non-persistable exchanges release chunks via
  `FilterAction::Release` to avoid holding
  pass-through traffic.

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
opts in for development. Host validation is re-run
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
- `apis/src/service/responses/`: independently testable Responses service
- `apis/src/service/conversations/`: independently testable Conversations service
- `store/`: SQL-free contracts, records, registries, and factory traits
- `store-lifecycle/`: cache, retry, generation leases, reuse, and retirement
- `store-backends/`: SQLx implementations, pools, schemas, TLS, and factories
- `server/src/store_provision.rs`: listener planning, readiness, and serving-runtime provisioning
- `server/src/reload.rs`: atomic pipeline-generation reload orchestration

## Related

- [AI Inference](ai-inference.md)
- [PostgreSQL cryptographic boundary](postgres-cryptographic-boundary.md)
- [Features](../features.md)
