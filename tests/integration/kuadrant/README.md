# Kuadrant MCP Gateway acceptance harness

Stands up a minimal, **no-full-Kubernetes** real Kuadrant MCP Gateway stack in
containers and proves — through Praxis's OpenAI Responses agentic loop — that the
gateway enforces per-user tool authorization (discovery filtering + per-call
authorization). The Praxis process uses the canonical
`examples/configs/openai/responses/full-flow-agentic.yaml`, retargeted to local
fixtures. A deterministic passing NeMo endpoint keeps the full-flow tool-result
guardrails active; the model and MCP backend are also deterministic fixtures.
See the design and plan under `docs/superpowers/`.

This is **acceptance** of already-delivered Praxis behavior: there are **no Praxis
product-code changes**.

## Stack (7 containers, one network `kuadrant-accept`)

| Container | Image (pinned) | Role |
|---|---|---|
| `envoy` | `docker.io/envoyproxy/envoy:v1.35.0` | data plane: `ext_proc`(router) → `ext_authz`(Authorino) → router |
| `mcp-broker` | `ghcr.io/kuadrant/mcp-gateway:v1.0.2` | broker (tools/list filter) + router (ext_proc) |
| `authorino` | `quay.io/kuadrant/authorino:v0.28.0` | JWT validation, wristband minting, per-call ext_authz |
| `kube-apiserver` | `registry.k8s.io/kube-apiserver:v1.33.6` | bare apiserver (CRD + Secret + AuthConfig); no operator/nodes |
| `etcd` | `quay.io/coreos/etcd:v3.5.17` | apiserver store |
| `jwks` | `python:3.12-slim` | static-JWKS OIDC issuer for per-user tokens |
| `weather-backend` | `python:3.12-slim` | fake MCP server with a `/stats` execution counter |

## Usage

```console
# bring the gateway stack up (engine: podman or docker; honors CONTAINER_ENGINE)
bash tests/integration/kuadrant/stack_up.sh          # writes .stack-env

# run the full acceptance (builds nothing; expects praxis-ai on PATH via PRAXIS_AI_BIN
# or target/{debug,release}/praxis-ai) — see the Makefile target:
make test-kuadrant-acceptance

# tear down
bash tests/integration/kuadrant/teardown.sh          # add --purge to drop .run/ + .stack-env
```

## Runtime & CI

Measured runtime (local, warm image cache): stack-up ~30s; the 18-case
buffered+streaming matrix ~2min (the item-6 live-revocation cases dominate — each
applies a revoked AuthConfig, confirms the reconcile, then restores). The praxis-ai
build (`make test-kuadrant-acceptance` rebuilds with `full,openai-all,store-sqlite`) adds a few
minutes cold. Budget ~10–15min on a cold CI runner (30min lane timeout). The gated
lane (`.github/workflows/kuadrant.yaml`) is path-filtered (only MCP courier / harness
/ example-config changes trigger it) and auto-registered as a required check by
`ci-status.yaml` — no central list edit.

The full-flow configuration also needs `WEB_SEARCH_API_KEY` at startup. The
acceptance supplies a dummy value because the model never requests web search.
Both Praxis processes use isolated temporary SQLite databases, and teardown
removes them.

`stack_up.sh` emits `.stack-env` (git-ignored) with `KUADRANT_GATEWAY_URL`,
`KUADRANT_ISSUER_PRIV`, `KUADRANT_ISSUER_KID`, `KUADRANT_KUBECONFIG`,
`KUADRANT_BACKEND_STATS`, consumed by the pytest acceptance
(`tests/integration/sdk/openai/test_openai_responses_mcp_kuadrant.py`).

## Requirements

podman **or** docker, `kubectl`, `openssl`, `uv` (PEP 723 python deps), `curl`.
All images are amd64-native and arm64-native. Generated PKI/keys/JWKS live under
`.run/` (git-ignored).

## Notes

- Broker verifies the ES256 `x-mcp-authorized` wristband from the env-only
  `TRUSTED_HEADER_PUBLIC_KEY`; `--enforce-capability-filtering` makes discovery
  fail-closed; `--mcp-gateway-private-host=envoy:8080` makes the router's
  `tools/call` hairpin reachable.
- Authorino uses `authentication.jwt.issuerUrl` (OIDC discovery) — **not** `jwksUrl`
  (which breaks wristband minting).
- `configs/authconfig-filtered.yaml` is the per-user discovery-filtering policy;
  `configs/authconfig-opendisco.yaml` exposes tools to all so per-call denial can be
  exercised independently (the "discovery ≠ enforcement" proof).
