# Local MCP filter POC

This is a development-only setup. It adds no production example config, Kubernetes resources, control-plane integration, passthrough, or non-list/call MCP methods. The `mcp_poc` Rust filter is opt-in behind `mcp-poc-filter`; its config and all service assets live here.

## Flow and filter boundary

```text
MCP classify (`mcp_poc`) → PPE (`policy`) → MCP response (`mcp_poc`) → router → load balancer
```

The existing `mcp` filter is not used in this POC. Its separate static-broker mode uses a config-authored catalog: the current profile can list tools but does not route `tools/call`; the stateless profile can route calls, but its catalog remains static and it does not pull upstream discovery.

The purpose-built `mcp_poc` filter appears twice with explicit `classify` and `respond` stages. The classifier validates requests in the bounded body pre-read, publishes `mcp.method`/`mcp.name`, and prepares routing for known tool calls. PPE runs between the stages and derives a request-scoped capability set from the static `mcp_authorization` policy extension and verified identity roles. The responder filters `tools/list` and checks calls against that same set before routing. Both instances reference the same catalog file, though each loads its own in-memory copy at pipeline build.

Discovery is an explicit operator action. The `discovery` Compose profile uses the official FastMCP client in modern auto mode, calls `tools/list` on both upstreams, refuses to emit a catalog unless both negotiate `2026-07-28`, and writes `catalog.json`. It never runs as part of gateway requests.

## Services

| Service | Version/source | Host listener |
| --- | --- | --- |
| Praxis | This checkout, workspace `0.11.0`; built with `mcp-poc-filter,opentelemetry` | `127.0.0.1:18080` |
| Keycloak | Official `quay.io/keycloak/keycloak:26.8.2`, realm `mcp-poc`, client/audience `praxis-gateway`, users `demo` / `demo` and `limited` / `limited` | `https://localhost:18443` |
| FastMCP | Official FastMCP SDK `4.2.0`, Python `3.13.9`, stateless HTTP | `127.0.0.1:18101` |
| Everything | Official `modelcontextprotocol/servers` source at `e1dc4b18dbb948921da37f3d4224a628730b43b0`, package `2.3.0`, official `src/everything/Dockerfile`, Streamable HTTP | `127.0.0.1:18102` |
| Tempo | Grafana Tempo `3.1.1`, local storage, OTLP gRPC | `127.0.0.1:4317` (ingest), `127.0.0.1:13200` (HTTP API) |
| Grafana | Grafana `12.4.15`, provisioned Tempo datasource | `http://127.0.0.1:13000` |

The Everything build uses the upstream repository root as context so its official Dockerfile can use the root npm lockfile. The upstream commit is fixed in `compose.yaml`; no published Everything image is used. Host ports are explicitly bound to loopback. Ports `8001` and `8002` were already allocated on this workstation, so the POC maps to `18101` and `18102` instead.

The FastMCP dependency closure is fully version-pinned in `requirements.txt`; the Python base image is pinned to `3.13.9`.

Keycloak creates a local PKCS#12 certificate inside its container. Only the exported public certificate is written to `certs/keycloak.crt`; the private key stays in the container. `SSL_CERT_FILE` is set only for the Praxis process, and all curl calls to Keycloak use `--cacert`. No machine-wide trust store is changed and TLS verification is not disabled.

`allow_private_idp` and `allow_private_endpoints` are enabled only in these POC configs for the loopback Keycloak and MCP endpoints. Do not copy those settings into production.

The PPE request assertion strips the inbound `Authorization` header before upstream forwarding. The validated Keycloak realm roles are used for gateway authorization only.

No OpenAI-compatible mock is included because this MCP path does not invoke an inference backend.

Tempo and Grafana are development-only. Their host ports bind to loopback. To add them without restarting the already-running Keycloak and MCP servers:

```console
docker compose -f dev/mcp-poc/compose.yaml up --detach tempo grafana
```

Grafana login: `mcp-poc` / `mcp-poc-grafana-local`. Tempo receives traces from the host Praxis process on `127.0.0.1:4317`; Grafana queries it over the Compose network.

## Start and discover

Run from the checkout root:

```console
docker compose -f dev/mcp-poc/compose.yaml up --build --detach
```

Wait for Keycloak's generated public certificate, then verify its HTTPS realm metadata:

```console
until test -s dev/mcp-poc/certs/keycloak.crt; do sleep 1; done
curl --fail-with-body --show-error \
  --cacert dev/mcp-poc/certs/keycloak.crt \
  https://localhost:18443/realms/mcp-poc/.well-known/openid-configuration
```

Pull the tools from both servers and write the catalog consumed by `mcp_poc`:

```console
docker compose --profile tools -f dev/mcp-poc/compose.yaml \
  run --build --rm discovery
```

This command requires both upstreams to negotiate MCP `2026-07-28`. If either server does not, discovery exits with an error instead of writing a misleading catalog. Re-run it after changing upstream tools; then restart Praxis to load the new file.

## Build and run Praxis from this checkout

Use a dedicated Cargo target directory to avoid accidentally running an existing local binary. The `full` default includes PPE; `mcp-poc-filter` enables the POC filter, and `opentelemetry` enables Praxis core OTLP export plus AI semantic spans.

The PPE capability projection is in the companion `../praxis` checkout. Check out the corresponding Praxis branch there, then run `make patch-praxis` before building. Run `make unpatch-praxis` to restore the registry dependencies afterward.

```console
CARGO_TARGET_DIR=target/mcp-poc-final \
  cargo build -p praxis-ai-proxy --features mcp-poc-filter,opentelemetry

SSL_CERT_FILE="$PWD/dev/mcp-poc/certs/keycloak.crt" \
CARGO_TARGET_DIR=target/mcp-poc-final \
  cargo run -p praxis-ai-proxy --features mcp-poc-filter,opentelemetry -- \
  -c dev/mcp-poc/praxis.yaml
```

For a config-only check while services are running:

```console
SSL_CERT_FILE="$PWD/dev/mcp-poc/certs/keycloak.crt" \
CARGO_TARGET_DIR=target/mcp-poc-final \
  cargo run -p praxis-ai-proxy --features mcp-poc-filter,opentelemetry -- \
  --validate -c dev/mcp-poc/praxis.yaml
```

OTLP exporting is initialized when Praxis starts. After building with the features above, stop the currently running Praxis process with Ctrl-C and rerun it with this config. Keycloak and both MCP servers can stay running.

## Mint a test token

```console
set -euo pipefail
export KC_CA="$PWD/dev/mcp-poc/certs/keycloak.crt"
TOKEN="$(curl --fail-with-body --silent --show-error \
  --cacert "$KC_CA" \
  --request POST \
  --header 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'grant_type=password' \
  --data-urlencode 'client_id=praxis-gateway' \
  --data-urlencode 'username=demo' \
  --data-urlencode 'password=demo' \
  'https://localhost:18443/realms/mcp-poc/protocol/openid-connect/token' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
export TOKEN
```

The client has an audience mapper for `praxis-gateway`; PPE validates the signature against the realm JWKS, issuer, and audience. The password grant and `demo` credentials are for local testing only.

## Direct upstream checks

These requests use the stateless profile's `Mcp-Protocol-Version`, `Mcp-Method`, `Mcp-Name` where applicable, and full per-request `_meta` envelope (protocol version, client info, client capabilities). They bypass Praxis and do not require OIDC.

```console
set -euo pipefail
export MCP_META='{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"poc-curl","version":"1.0"},"io.modelcontextprotocol/clientCapabilities":{}}'

# FastMCP tools/list
curl --fail-with-body --show-error http://127.0.0.1:18101/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/list' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{\"_meta\":$MCP_META}}"

# Official Everything tools/list
curl --fail-with-body --show-error http://127.0.0.1:18102/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/list' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{\"_meta\":$MCP_META}}"

# FastMCP echo
curl --fail-with-body --show-error http://127.0.0.1:18101/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: echo' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"echo\",\"arguments\":{\"message\":\"direct-fastmcp\"},\"_meta\":$MCP_META}}"

# Official Everything get-sum
curl --fail-with-body --show-error http://127.0.0.1:18102/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: get-sum' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"get-sum\",\"arguments\":{\"a\":19,\"b\":23},\"_meta\":$MCP_META}}"
```

## Through Praxis

```console
# Modern client discovery advertises the supported stateless version.
curl --fail-with-body --show-error http://127.0.0.1:18080/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: server/discover' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"server/discover\",\"params\":{\"_meta\":$MCP_META}}"

# The POC catalog lists prefixed tools from both upstreams.
curl --fail-with-body --show-error http://127.0.0.1:18080/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/list' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/list\",\"params\":{\"_meta\":$MCP_META}}"

# Route fastmcp_echo to FastMCP; the gateway strips fastmcp_ before forwarding.
curl --fail-with-body --show-error http://127.0.0.1:18080/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: fastmcp_echo' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{\"name\":\"fastmcp_echo\",\"arguments\":{\"message\":\"through-fastmcp\"},\"_meta\":$MCP_META}}"

# Route everything_get-sum to the official Everything server; the gateway strips everything_.
curl --fail-with-body --show-error http://127.0.0.1:18080/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' \
  -H 'Mcp-Name: everything_get-sum' \
  --data-binary "{\"jsonrpc\":\"2.0\",\"id\":13,\"method\":\"tools/call\",\"params\":{\"name\":\"everything_get-sum\",\"arguments\":{\"a\":19,\"b\":23},\"_meta\":$MCP_META}}"
```

The distinct result content (`fastmcp-echo: ...` versus `The sum of 19 and 23 is 42.`) plus the selected cluster in the Praxis request path demonstrates prefix routing. The FastMCP container also exposes a loopback-only POC counter at `http://127.0.0.1:18101/__poc/metrics`; it counts POSTs to its MCP endpoint for the authorization checks below.

## Tool authorization

`dev/mcp-poc/policy.yaml` is the single static authorization source for list visibility and calls. Praxis's `PolicyFilter` reads the host-specific `mcp_authorization` extension, maps the validated Keycloak realm roles from `claim_mapper: keycloak` into a typed request-scoped `McpAllowedCapabilities`, and removes the extension before giving the remaining document to the APL engine. This POC uses no CEL.

```yaml
mcp_authorization:
  role_grants:
    mcp-tools-all:
      - server: "*"
        tool: "*"
    mcp-tool-fastmcp-echo:
      - server: fastmcp
        tool: echo
```

The stable authorization key is `(server ID, backend tool name)`, not the public prefixed name. `McpPocFilter::tools_list` intersects the catalog with the PPE-produced capability set. Its post-PPE call path checks the same set before the router runs. Missing capabilities fail closed; a denied direct call returns a JSON-RPC error with code `-32003` and does not reach the upstream.

The imported `demo` user has `mcp-tools-all`. The `limited` user has only `mcp-tool-fastmcp-echo`; its list contains only `fastmcp_echo`, and a direct `everything_get-sum` call is denied even if the client skips the list.

## Trace inspection

The POC config sets `sampling_rate: 1.0`, sampling every new/root request; core honors the sampled flag on an incoming W3C parent context. No special request header is needed to enable export. `smoke.sh` adds sampled `traceparent` headers to the demo gateway `tools/list` and `tools/call` requests and the restricted-user `tools/list` request. It waits for those traces to appear in Tempo and prints labeled trace IDs. Other requests without a parent context are sampled as roots.

Open Grafana at `http://127.0.0.1:13000` and sign in with `mcp-poc` / `mcp-poc-grafana-local`. Go to **Explore**, select the provisioned **Tempo** datasource, and search by one of the trace IDs printed by `smoke.sh`.

A local `tools/list` request produces three spans: the inbound `POST` server span, `mcp_poc.classify`, and `mcp_poc.respond`. An authorized forwarded `tools/call` produces five: those three plus Praxis core’s `POST` client span and `upstream_exchange`. A denied direct call stops in the POC response stage and does not create the two upstream spans.

The MCP spans record bounded method labels (`server/discover`, `tools/list`, `tools/call`, or `other`) and outcomes (`route`/`respond`, `forward`/`deny`/`local_response`). For a valid catalog-routed call, expand `mcp_poc.classify` to see `mcp.tool` (the public prefixed name, such as `fastmcp_echo`) and `mcp.backend.tool` (the upstream name, such as `echo`). Both names come from the validated catalog; they are not copied directly from the request. The spans never record JSON-RPC IDs, request bodies, arguments, credentials, or results. PPE has no separate decision span. Direct calls to MCP servers bypass Praxis and are not in these traces.

Expand `mcp_poc.respond` in the restricted-user `tools/list` trace to see the authorization event with `operation=tools/list`, `tools_before=24`, and `tools_after=1`. Full-access list requests record the same event with equal counts.

Tempo’s loopback-only HTTP API is also available at `http://127.0.0.1:13200/api/traces/<trace-id>`.

## Access-control checks and backend evidence

`verify-auth-denials.sh` verifies that missing and invalid tokens receive 401 responses without backend traffic. It also gets a token for `limited`, checks that only `fastmcp_echo` is listed, verifies that its FastMCP call succeeds, and confirms a direct call to hidden `everything_get-sum` is denied without increasing the Everything POST count.

Run the complete live request flow, including direct upstream calls, prefixed calls through Praxis, credential and restricted-user authorization checks, then wait for the sampled traces in Tempo and verify Prometheus has scraped Praxis request metrics:

```console
./dev/mcp-poc/smoke.sh
```

The script requires the Compose services, a current `catalog.json`, and Praxis running from this checkout using the source-run command above. Run only the authorization checks with:

```console
./dev/mcp-poc/verify-auth-denials.sh
```
## Rebuild and stop

```console
# Rebuild Python service/discovery image and the pinned official Everything source image.
docker compose -f dev/mcp-poc/compose.yaml build --no-cache fastmcp everything
docker compose -f dev/mcp-poc/compose.yaml up --force-recreate --detach fastmcp everything
# Recreate Keycloak after changing the local realm or certificate bootstrap.
docker compose -f dev/mcp-poc/compose.yaml up --force-recreate --detach keycloak
# Stop and remove the local containers/network; the images remain cached.
docker compose -f dev/mcp-poc/compose.yaml down
```

## Compatibility and references

Verified in this checkout: FastMCP `4.2.0` and the pinned Everything server `2.3.0` both negotiated MCP `2026-07-28`; direct `tools/list` and deterministic `tools/call` requests succeeded for each. The discovery utility wrote 24 tools, and Praxis returned all 24 with prefixes. Calls to `fastmcp_echo` and `everything_get-sum` returned results from their respective backends. Missing and invalid OIDC credentials each returned HTTP 401 for both routes; FastMCP's `/mcp` POST counter and Everything's POST log count were unchanged across the denied calls.

The POC intentionally implements only `server/discover`, `tools/list`, and `tools/call`. Refreshing the catalog is manual; sessions, other MCP methods, and passthrough are out of scope. The Everything reference listing also includes tools that exercise tasks, elicitation, and other unsupported capabilities; only `get-sum` was called here. The existing `mcp` static broker remains a baseline, not the POC filter.

The pinned Everything source commit builds through its official Dockerfile and root npm lockfile. That Dockerfile still uses the mutable `node:24-alpine` base tag, and `npm ci` reported two moderate audit advisories in the upstream dependency tree; no upstream files or dependencies were changed.

Official implementation references:

- [FastMCP HTTP deployment and stateless mode](https://gofastmcp.com/docs/deployment/http)
- [FastMCP client protocol negotiation](https://gofastmcp.com/docs/clients/client)
- [Official Everything Dockerfile](https://github.com/modelcontextprotocol/servers/blob/e1dc4b18dbb948921da37f3d4224a628730b43b0/src/everything/Dockerfile)
- [Official Everything modern Streamable HTTP handler](https://github.com/modelcontextprotocol/servers/blob/e1dc4b18dbb948921da37f3d4224a628730b43b0/src/everything/transports/streamableHttp.ts)
- [Keycloak TLS configuration](https://www.keycloak.org/server/enabletls)
- [Keycloak startup realm import](https://www.keycloak.org/server/importExport)
