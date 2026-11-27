# Agentic protocols

How Praxis AI parses JSON-RPC 2.0 envelopes and extracts protocol metadata for
MCP and A2A agent traffic. Per-filter configuration lives on the generated
[`mcp`](../filters/agents/mcp.md) and [`a2a`](../filters/agents/a2a.md) pages.

## Overview

Three filters form a layered stack. Each parses the JSON-RPC envelope from the
request body and promotes protocol metadata to headers, metadata, and filter
results.

```text
                    JSON-RPC 2.0 Envelope
                           |
              +------------+------------+
              |                         |
           MCP Filter              A2A Filter
              |                         |
     +--------+--------+       +-------+-------+
     |                  |       |               |
  Broker           Tool/       Task          SSE
  (catalog,      Resource     Routing      Scanner
   session)      metadata                (streaming)
```

All three use `BodyAccess::ReadOnly` with `BodyMode::StreamBuffer` and a
configurable `max_body_bytes`. Promoted header and metadata values are limited
to 256 bytes and checked for control characters. `on_invalid` sets what happens
when the body is not valid JSON-RPC (`continue`, `reject`, or `error`). The
agentic filters are always compiled, with no feature flag.

## JSON-RPC foundation

The `json_rpc` filter extracts envelope metadata from HTTP POST bodies:

- Kind: Request, Notification, Response, or Batch
- Method: the `method` string field
- ID: the `id` field (string, integer, or null)
- Batch length: element count for batch requests

The MCP and A2A filters call `parse_json_rpc_value` as their first parsing step
and add protocol-specific extraction on top.

### Configuration

- `max_body_bytes`: maximum body size to parse
  (default 1 MiB)
- `batch_policy`: `reject` (default) or `first`
  (extract metadata from the first element)
- `on_invalid`: `continue` (default), `reject`
  (400), or `error` (JSON-RPC error response)
- Header names are configurable (default
  `X-Json-Rpc-Method`, `X-Json-Rpc-Id`,
  `X-Json-Rpc-Kind`)

## MCP protocol

The `mcp` filter extracts Model Context Protocol metadata and can act as a tool
catalog broker.

### Metadata extraction

From the JSON-RPC envelope, the filter extracts:

- Method: known MCP methods (`initialize`, `tools/list`, `tools/call`,
  `resources/read`, `prompts/get`, current/legacy `ping`, etc.) plus
  `Other(String)` for extensions
- Name: tool, resource, or prompt name from `params` (for methods like
  `tools/call`)
- Protocol version: from `initialize` params or the `Mcp-Protocol-Version`
  header
- Session ID: from the `Mcp-Session-Id` header

### Header validation

The filter checks `Mcp-Method` and `Mcp-Name` headers against the body-derived
values. On a mismatch it can `reject` or `ignore`; on a missing header it can
`ignore`, `synthesize`, or `reject`.

### Broker architecture

When the config has a `servers` key, the filter acts as an MCP broker over
one catalog of tools from several servers:

```text
Client --> MCP Broker Filter
             |
             +-- initialize: session + version
             |                negotiation
             +-- tools/list:  the caller's catalog
             |                from all servers
             +-- ping:        local response (current profile only)
             +-- tools/call:  routes by configured
             |                catalog tool; current profile
             |                needs backend_sessions, else
             |                returns -32601
             +-- DELETE:      session termination (and
                              backend session cleanup)
```

Each server entry has a name, upstream cluster, path prefix, tool prefix (for
namespace isolation), and tool list. `tools/list` returns the tools of all
configured servers as one catalog.

### Tool discovery

A server with `discover: true` contributes the tools its own `tools/list`
returns instead of only its static `tools:`. Servers can show different tools
to different callers, so the catalog is per caller:

- The broker lists a server's tools with the caller's forwarded credential:
  `backend_sessions.forward_headers` in the current profile and
  `discovery.forward_headers` in the stateless profile. The listing is cached
  per caller and server, keyed on the fingerprint of those headers, so a
  refreshed token is a new caller until #2191 binds callers to a verified
  identity.
- In the current profile the listing runs inside the caller's backend session
  for that server, so a later `tools/call` reuses the session. A `tools/list`
  therefore opens backend sessions for discovering servers that have none yet,
  and requires a gateway `Mcp-Session-Id` (HTTP `400` without one). A backend
  `404` during the listing reopens the session once. The stateless profile
  sends `tools/list` without a session and marks the result
  `cacheScope: private`.
- The broker follows `nextCursor` for at most 100 pages, `discovery.max_tools`
  tools, and 4 MiB of tool definitions, the bounds the hosted MCP connectors
  use.
- Servers that need a listing are discovered concurrently, at most
  `discovery.max_concurrency` at a time, and share one budget:
  `backend_sessions.handshake_timeout_ms` (handshake and pages together) in the
  current profile, `discovery.timeout_ms` in the stateless profile.
- A `tools/call` discovers only when the caller's cached catalog does not
  expose the name. A name that left the listing gets `unknown tool`.

Each listed tool is vetted before any client sees it. A tool needs a name and
an object `inputSchema`; an invalid tool is left out, or fails the server's
listing under `invalid_tool_policy: reject_server`. A name listed twice fails
the listing. Operator admission then applies the server's `tool_policy`, with
the rule `tool_admission` applies on the hosted paths: `allow_all` (the
default) admits every tool, and `read_only` admits only tools annotated
`readOnlyHint: true` without `destructiveHint: true`.

A static `tools:` entry with the name of a discovered tool overrides its
description, or hides it with `hidden: true`. The server's own schema and
annotations are kept. A static entry the server does not list is still
exposed. Exposed names keep `tool_prefix`. When a server exposes a name an
earlier server in `servers:` already exposes, the later server is left out of
that caller's catalog and the collision is logged with both server names.

A listing older than `discovery.max_age_ms` is rediscovered on the caller's
next request. When discovery fails (timeout, transport error, error status,
JSON-RPC error, or a bound), the caller keeps its last good listing for that
server, or the server's static entries when it has none. The other servers are
unaffected. Each failure is logged and counted in
`praxis_ai_mcp_broker_discovery_failures_total{server, reason}`. The cache
holds at most `discovery.max_entries` listings, evicting the least recently
read, and drops a listing no request has read for `discovery.idle_timeout_ms`.
It is per filter instance, so a reload starts empty. Refresh on
`notifications/tools/list_changed` and dropping tools of unhealthy servers are
tracked in #2191.

### Current-profile backend sessions

Stateful MCP backends reject `tools/call` without a prior `initialize` on a
backend session. With `backend_sessions` configured, the broker maps each
`(gateway Mcp-Session-Id, server)` pair to one backend session:

```text
tools/call (gateway session G, tool weather_x)
  |
  +-- mapping (G, weather) missing?
  |     initialize            --> weather cluster
  |     capture Mcp-Session-Id B
  |     notifications/initialized (B) --> weather cluster
  |
  +-- forward tools/call with Mcp-Session-Id B
  |   through the main chain (streams, policies apply)
  |
  +-- backend 404 --> drop (G, weather); the next call
                      runs a fresh handshake
```

#### Handshake

The handshake runs as a filtered subrequest through a generated outbound chain:
the `mcp_broker_backend` selector, any `outbound_filters`, and a
`load_balancer` that references each server cluster by name. Server clusters
must therefore be declared in top-level `clusters:`. Each exchange goes through
the MCP client the hosted MCP paths use (`praxis_ai_apis::mcp_session`): `rmcp`
builds and decodes the JSON-RPC messages. No `rmcp` service is kept between
exchanges, so an idle backend session holds no task and no standalone `GET`
stream. Each reply is bounded by `max_response_bytes`: a decoded body over the
limit is rejected before it is fully retained, whatever its transfer framing. An `initialize` reply on
an SSE stream completes when its JSON-RPC response arrives, even if the backend
keeps the stream open, unless an `outbound_filters` entry buffers response
bodies.

A backend may send requests on the `initialize` stream before its result and
wait for the answers. The broker answers `ping` with an empty result and any
other request with `-32601`, each in a new POST through the same chain with the
forwarded headers and the issued `Mcp-Session-Id`. It sends no
`MCP-Protocol-Version`, because no version is negotiated yet. It answers at most
four requests per handshake. A rejected or undelivered answer fails the
handshake. With a buffering `outbound_filters` entry, answers go out only after
the stream ends, so a backend that waits for one times out.

`handshake_timeout_ms` is one budget for the whole handshake. It starts before
the broker waits on a concurrent handshake for the same pair, time spent in
`initialize` is taken from `notifications/initialized`, and a parent
agent-loop deadline can shorten it. Each background session termination gets a
fresh budget of the same length.

A backend session exists only on the endpoint that issued it. The broker
records the endpoint the handshake's `load_balancer` dialed and pins
answers to server requests on the `initialize` stream,
`notifications/initialized`, session `DELETE`, and every forwarded call to it
through core's `pinned_endpoint_address`. The cluster's TLS and connection
settings still apply, and on a multi-endpoint cluster each session stays on one
replica. A `502` from the pinned endpoint drops the mapping like a `404`. So
does a proxy `502` or `504` for a call that got no backend response at all
(refused connection, reset, TLS failure, or timeout), so the next call
initializes on a reachable replica. Sessionless backends are not pinned.

If a backend answers `initialize` with a protocol version outside the broker's
`supported_versions`, the broker never confirms the session: it terminates it
and returns an `unsupported_protocol_version` error instead of forwarding the
call. A session the backend issued in an `initialize` that then failed is
terminated the same way, without an `MCP-Protocol-Version`. A client
`MCP-Protocol-Version` the broker does not support gets HTTP 400 before any
handshake; without the header the broker offers its default version. SSE replies go through the shared conforming parser (CR, LF, or CRLF
line endings and an optional leading BOM). As the SSE specification requires,
an event the backend never terminated with a blank line is discarded.

Concurrent first calls for one pair share a single handshake. A mapping is
bound to a SHA-256 fingerprint of the forwarded credential headers, so a
different credential on the same gateway session runs its own handshake and the
session it replaces is terminated. A backend `404` drops the mapping even while
another handshake for the pair is running.

#### Session map limits

The map is in memory, per filter instance, capped by `max_sessions` (least
recently used entries are evicted) and expired after `idle_timeout_ms`. Idle
time counts from the end of the last call that used the mapping, and idle
entries are swept on traffic at most once per `idle_timeout_ms`. A mapping in
use by an in-flight handshake or forwarded call is never evicted or expired
until that request completes. A credential change can still replace it, so a
call that started earlier may finish on the replaced session while new calls
use the new one. When every mapping is in use, a call for a new pair fails with
`session_capacity`.

Evicted, expired, and credential-replaced sessions are terminated in the
background with the credentials that created them, which the broker keeps in
memory (marked sensitive) for the session's lifetime. Termination is best
effort: at most 16 run at once. A session whose termination is skipped because
all 16 are busy, fails, or is lost to a restart or reload is left to the
backend's own idle expiry.

#### Gateway session ids

With `backend_sessions`, the broker accepts only gateway `Mcp-Session-Id`
values it issued through `initialize` and that no `DELETE` has ended. An id
carries an issue sequence number, random bytes, and their HMAC-SHA256 tag under
a key drawn from OpenSSL when the filter is built, so checking it needs no
per-session state. Any request with another well-formed id, whatever its
method (`initialize` and `GET` included), gets HTTP `404` and opens no backend
session; the client must send a new `initialize` without the id. A malformed id
gets `400`. A `GET` with a live id keeps its `405`. A config reload or restart
draws a new key, so every outstanding id gets `404` and clients reinitialize.
Without `backend_sessions`, ids are not signed or checked.

Ended ids are kept in a list of at most 16384 entries per filter instance. When
it is full, the lowest entry is dropped and every id issued up to it is
rejected from then on, live or ended. Forgetting state can end ids but never
revive one. A client whose live id expires this way gets `404` and
reinitializes.

#### `DELETE`

`DELETE` of an unknown or ended id is a `404`. While the gateway session has
backend sessions or handshakes created with credentials other than the
`DELETE`'s, the `DELETE` terminates only the caller's own backend sessions
(including one whose handshake is still in flight) and answers `409`. The
gateway session stays usable, because presenting the id alone must not end a
session another caller is using. Otherwise the `DELETE` ends the id and answers
`204`.

The decision and the end happen under the session map lock, which every
backend handshake also takes before it starts, so a handshake either starts
before the `DELETE` sees it or never starts. A `tools/call` waiting on a
handshake when the id ends gets `404` without contacting the backend. A
handshake the `DELETE` cancelled terminates its new backend session instead of
publishing it (best effort: if that fails, the session is left to the
backend's idle expiry), and its call fails with `session_closed`.

Ownership follows the forwarded credential headers, not a caller identity. A
client whose token was refreshed, or a `policy` filter that mints a new
delegated token before the broker, presents a different credential. A `DELETE`
of its own gateway session then answers `409`, and backend sessions created
with the earlier token are left to idle expiry. Binding sessions to an asserted
identity instead of the credential is tracked in #2191.

Several broker replicas behind one gateway address are not supported yet. The
signing key, the ended-id list, and the backend mappings live in one broker
instance, so clients must reach the same replica for a session's lifetime.

### Filter results

Written under key `"mcp"` with fields: `method`,
`name`, `protocol_version`, `session_present`,
`kind`.

## A2A protocol

The `a2a` filter extracts Agent-to-Agent protocol metadata, with optional
task-ownership routing.

### Metadata extraction

From the JSON-RPC envelope, the filter extracts:

- Method: 11 known A2A methods (`SendMessage`, `SendStreamingMessage`,
  `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, push notification
  config methods, `GetExtendedAgentCard`) plus `Unknown(String)`
- Family: `Message`, `Task`, `PushNotification`, `AgentCard`, or `Unknown`
- Streaming: whether the method produces SSE
- Task ID: from request params (for task methods) or response bodies
- Version: from the `A2A-Version` header
- Method aliases: configurable mapping for alternative method names (e.g.
  slash-style to PascalCase)

### Task routing

When enabled, the filter keeps an in-process task-to-cluster map and routes
follow-up requests to the agent that owns the task:

```text
1. SendMessage --> upstream A
     response contains task_id: "t1"
     store: t1 -> cluster A

2. GetTask(t1) --> route to cluster A
     (looked up from store)

3. CancelTask(t1) [terminal] --> cluster A
     remove: t1 from store (after TTL)
```

`LocalTaskRouteStore` is a `RwLock<HashMap>` with TTL expiry. Non-terminal
task states default to a 3600s TTL; terminal states (`completed`, `failed`,
`canceled`, `rejected`) default to 300s.

### Response body processing

The filter captures task routes from response bodies on two paths:

- JSON responses: hex-encoded buffering in `filter_metadata`, parsed at
  end-of-stream.
- SSE responses (streaming methods): `SseScanState` scans incrementally and
  extracts routes per event. It handles arbitrary chunk boundaries,
  CRLF/LF/CR line endings, multi-line `data:` fields, and scratch buffer
  overflow.

### Filter results

Written under key `"a2a"` with fields: `method`,
`family`, `streaming`, `kind`, `task_id`, `version`.

## Key files

- `filters/src/agentic/mcp/`:
  MCP filter, broker, envelope, protocol
- `filters/src/agentic/mcp/broker/`:
  MCP broker with static catalog
- `filters/src/agentic/a2a/`:
  A2A filter, task routing, SSE scanner
- `filters/src/agentic/a2a/task_routing.rs`:
  in-process task route store with TTL

## Related

- [AI Inference](ai-inference.md)
- [Response Store](response-store.md)
- [Features](../features.md)
