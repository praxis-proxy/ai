# Agentic Protocols

JSON-RPC 2.0 envelope parsing and protocol-specific
metadata extraction for MCP and A2A agent traffic.

## Overview

Three filters form a layered stack for agentic
protocol support. Each parses the JSON-RPC envelope
from request bodies and promotes protocol-specific
metadata to headers, metadata, and filter results.

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

## JSON-RPC Foundation

The `json_rpc` filter extracts JSON-RPC 2.0 envelope
metadata from HTTP POST request bodies:

- **Kind**: Request, Notification, Response, or Batch
- **Method**: the `method` string field
- **ID**: the `id` field (string, integer, or null)
- **Batch length**: element count for batch requests

Both MCP and A2A filters reuse `parse_json_rpc_value`
as their first parsing step, then layer
protocol-specific extraction on top.

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

## MCP Protocol

The `mcp` filter extracts Model Context Protocol
metadata and optionally acts as a tool catalog
broker.

### Metadata Extraction

From the JSON-RPC envelope, the filter extracts:

- **Method**: known MCP methods
  (`initialize`, `tools/list`, `tools/call`,
  `resources/read`, `prompts/get`, current/legacy
  `ping`, etc.)
  plus `Other(String)` for extensions
- **Name**: tool, resource, or prompt name from
  `params` (for methods like `tools/call`)
- **Protocol version**: from `initialize` params or
  `Mcp-Protocol-Version` header
- **Session ID**: from `Mcp-Session-Id` header

### Header Validation

The filter validates `Mcp-Method` and `Mcp-Name`
headers against body-derived values. Configurable
behavior on mismatch (`reject` or `ignore`) and on
missing headers (`ignore`, `synthesize`, or
`reject`).

### Broker Architecture

When the `servers` key is present in config, the
filter activates as an MCP broker with a static
tool catalog:

```text
Client --> MCP Broker Filter
             |
             +-- initialize: session + version
             |                negotiation
             +-- tools/list:  aggregated catalog
             |                from all servers
             +-- ping:        local response (current profile only)
             +-- tools/call:  routes by configured
             |                catalog tool; current profile
             |                needs backend_sessions, else
             |                returns -32601
             +-- DELETE:      session termination (and
                              backend session cleanup)
```

Each server entry specifies a name, upstream cluster,
path prefix, tool prefix (for namespace isolation),
and tool list. The broker aggregates tools from all
configured servers into a single catalog response.

### Current-Profile Backend Sessions

Stateful MCP backends reject `tools/call` without a
prior `initialize` on a backend session. With
`backend_sessions` configured, the broker maps each
`(gateway Mcp-Session-Id, server)` pair to one backend
session:

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

The handshake runs as a filtered subrequest through a
generated outbound chain: the `mcp_broker_backend`
selector, any `outbound_filters`, and a `load_balancer`
that references each server cluster by name, so server
clusters must be declared in top-level `clusters:`.
Each exchange goes through the MCP client the hosted MCP
paths use (`praxis_ai_apis::mcp_session`): `rmcp` builds
the messages and parses the replies, and the filtered
subrequest transport carries them. No `rmcp` service is
kept between exchanges, so an idle backend session costs
no task and opens no standalone `GET` stream.
Each reply is bounded by `max_response_bytes`; oversized
or chunked replies are rejected before the full body is
retained. An `initialize` reply on an SSE stream
completes as soon as its JSON-RPC response arrives, even
when the backend keeps the stream open; an
`outbound_filters` entry that buffers response bodies
turns that back into a read to end of stream.
`handshake_timeout_ms` is one budget for the
whole handshake, not for each exchange: it starts before
the broker waits on a concurrent handshake for the same
pair, time `initialize` spends is no longer available to
`notifications/initialized`, and a parent agent-loop
deadline can shorten it. Each background session
termination gets a fresh budget of the same length.

A backend session exists only on the endpoint that
issued it, so the broker records the endpoint the
handshake's `load_balancer` dialed and pins
`notifications/initialized`, session `DELETE`, and
every forwarded call to it through core's
`pinned_endpoint_address` (the cluster's TLS and
connection settings still apply). Clusters with
several endpoints therefore keep each session on one
replica. A `502` from the pinned endpoint drops the
mapping like a `404`, and so does a proxy `502` or
`504` for a call that got no backend response at all
(refused connection, reset, TLS failure, or timeout),
so the next call initializes on a reachable replica.
Sessionless backends are not pinned.

A backend that answers `initialize` with a protocol
version outside the broker's `supported_versions` is
never confirmed: the broker terminates that session and
returns a `unsupported_protocol_version` error instead
of forwarding the call. A client `MCP-Protocol-Version`
the broker does not support is rejected with HTTP 400
before any handshake; without the header the broker
offers its default version. SSE replies are decoded with
the shared conforming parser (CR, LF, or CRLF line
endings and an optional leading BOM); as the SSE
specification requires, an event the backend never
terminated with a blank line is discarded.

Concurrent first calls for one pair share a single
handshake. A mapping is bound to a SHA-256 fingerprint
of the forwarded credential headers, so a different
credential on the same gateway session runs its own
handshake, and the session it replaces is terminated. A
gateway `DELETE` terminates only the backend sessions
created with the caller's own credentials, including one
whose handshake is still in flight: that handshake
terminates its session instead of publishing it, and its
call fails with `session_closed`. A backend `404` drops
the mapping even while another handshake for the pair is
running.

The map is in memory, per filter instance, capped by
`max_sessions` (least recently used entries are evicted)
and expired after `idle_timeout_ms`. A mapping in use by
an in-flight handshake or forwarded call is never
evicted or expired until that request completes. A
credential change can still replace it, so a call that
started earlier may finish on the replaced session while
new calls use the new one. When every
mapping is in use, a call for a new pair fails with
`session_capacity`. Idle time counts from the end of the
last call that used the mapping. Evicted, expired, and
credential-replaced sessions are terminated in the
background with the credentials that created them, which
the broker keeps in memory (marked sensitive) for the
session's lifetime. Termination is best effort: at most 16
run at once, and a session whose termination is skipped
because all 16 are busy, fails, or is lost to a restart or
reload is left to the backend's own idle expiry. Idle
entries are swept on traffic, at most once per
`idle_timeout_ms`.

With `backend_sessions`, the broker accepts only gateway
`Mcp-Session-Id` values it issued through `initialize`
and that no `DELETE` has ended. An id carries an issue
sequence number, random bytes, and their HMAC-SHA256 tag
under a key drawn from OpenSSL when the filter is built,
so checking it needs no per-session state. Any request
carrying another well-formed id, whatever its method
(`initialize` and `GET` included), gets HTTP `404`, opens
no backend session, and the client must send a new
`initialize` without the id; a malformed id gets `400`.
A `GET` with a live id keeps its `405`.

Ended ids are kept in a list of at most 16384 entries per
filter instance. When it is full, the lowest entry is
dropped and every id issued up to it is rejected from then
on, live or ended: forgetting state can only end ids,
never revive one. A client whose live id expires this way
gets `404` and reinitializes.

`DELETE` of an unknown or ended id is a `404`. While the
gateway session has backend sessions or handshakes created
with other credentials than the `DELETE`'s, the `DELETE`
terminates only the caller's own backend sessions and
answers `409`; the gateway session stays usable, because
presenting the id alone must not end a session another
caller is using. Otherwise the `DELETE` ends the id and
answers `204`. The decision and the end happen under the
session map lock, which every backend handshake also takes
before it starts, so a handshake either starts before the
`DELETE` sees it or never starts: a `tools/call` waiting
on a handshake when the id ends gets `404` without
contacting the backend, and one whose handshake the
`DELETE` cancelled fails with `session_closed` after the
broker attempts to terminate the new backend session
(best effort: if that termination fails, the session is
left to the backend's idle expiry). A config reload or
restart draws a new key, so every outstanding id gets
`404` and clients reinitialize. Without
`backend_sessions`, ids are not signed or checked.

Ownership follows the forwarded credential headers, not a
caller identity. A client whose token was refreshed, or a
`policy` filter that mints a new delegated token before
the broker, presents a different credential, so a `DELETE`
of its own gateway session answers `409` while backend
sessions created with the earlier token remain; they are
left to idle expiry. Binding sessions to an asserted
identity instead of the credential is tracked in #2191.

Running several broker replicas behind one gateway
address is not supported yet: the signing key, the
ended-id list, and the backend mappings live in one
broker instance, so clients must reach the same replica
for a session's lifetime.

### Filter Results

Written under key `"mcp"` with fields: `method`,
`name`, `protocol_version`, `session_present`,
`kind`.

## A2A Protocol

The `a2a` filter extracts Agent-to-Agent protocol
metadata with optional task-ownership routing.

### Metadata Extraction

From the JSON-RPC envelope, the filter extracts:

- **Method**: 11 known A2A methods (`SendMessage`,
  `SendStreamingMessage`, `GetTask`, `ListTasks`,
  `CancelTask`, `SubscribeToTask`, push notification
  config methods, `GetExtendedAgentCard`) plus
  `Unknown(String)`
- **Family**: `Message`, `Task`,
  `PushNotification`, `AgentCard`, or `Unknown`
- **Streaming**: whether the method produces SSE
- **Task ID**: from request params (for task
  methods) or response bodies
- **Version**: from `A2A-Version` header
- **Method aliases**: configurable mapping for
  alternative method names (e.g. slash-style to
  PascalCase)

### Task Routing

When enabled, the filter maintains an in-process
task-to-cluster mapping for routing follow-up
requests to the agent that owns a task:

```text
1. SendMessage --> upstream A
     response contains task_id: "t1"
     store: t1 -> cluster A

2. GetTask(t1) --> route to cluster A
     (looked up from store)

3. CancelTask(t1) [terminal] --> cluster A
     remove: t1 from store (after TTL)
```

The `LocalTaskRouteStore` uses `RwLock<HashMap>`
with TTL-based expiry. Non-terminal task states
default to 3600s TTL; terminal states (`completed`,
`failed`, `canceled`, `rejected`) default to 300s.

### SSE Response Scanning

For streaming A2A methods, the filter scans response
bodies incrementally using `SseScanState`. The
scanner handles arbitrary chunk boundaries,
CRLF/LF/CR line endings, multi-line `data:` fields,
and scratch buffer overflow. Extracted task routes
from SSE payloads are written to the task route
store.

### Response Body Processing

The A2A filter captures response bodies to extract
task routes via two paths:

- **JSON responses**: hex-encoded buffering in
  `filter_metadata`, parsed at end-of-stream
- **SSE responses**: incremental line scanning via
  `SseScanState`, routes extracted per event

### Filter Results

Written under key `"a2a"` with fields: `method`,
`family`, `streaming`, `kind`, `task_id`, `version`.

## Shared Patterns

All three filters share these conventions:

- **Body access**: `BodyAccess::ReadOnly` +
  `BodyMode::StreamBuffer` with configurable
  `max_body_bytes`
- **Value length limit**: 256 bytes for any
  promoted header or metadata value
- **Control character check**: values are validated
  before promotion
- **`on_invalid` policy**: configurable behavior
  when the request body is not valid JSON-RPC
  (`continue`, `reject`, or `error`)
- **Feature gating**: agentic filters are always
  compiled (not behind a feature flag)

## Key Files

- `filters/src/agentic/mcp/`:
  MCP filter, broker, envelope, protocol
- `filters/src/agentic/a2a/`:
  A2A filter, task routing, SSE scanner
- `filters/src/agentic/mcp/broker/`:
  MCP broker with static catalog
- `filters/src/agentic/a2a/task_routing.rs`:
  in-process task route store with TTL

## Related

- [AI Inference](ai-inference.md)
- [Response Store](response-store.md)
- [Features](../features.md)
