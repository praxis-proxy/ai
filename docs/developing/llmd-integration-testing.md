# llm-d Integration Testing

## Overview

Praxis AI includes an AI-owned `ext_proc` compatibility layer for
[`llm-d`](https://github.com/llm-d/llm-d). It replaces the abandoned
Praxis-side crate publication approach (Praxis PR #780).

## Crate location and feature

- **Crate path:** `integrations/llmd/ext-proc/`
- **Package name:** `praxis-ai-llmd-ext-proc`
- **Feature name:** `llmd-ext-proc`
- **`publish = false`**: not available on crates.io.

Default `praxis-ai-proxy` builds do not link, register, or expose the
`ext_proc` filter. It is registered only when `praxis-ai-proxy` is built with
the `llmd-ext-proc` feature.

## Architecture

The integration test request path is:

```text
OpenAI-compatible request
    -> Praxis AI runtime
    -> ext_proc filter (gRPC full-duplex)
    -> deterministic mock ExternalProcessor
    -> trusted x-gateway-destination-endpoint mutation
    -> endpoint_selector
    -> llm-d-inference-sim container
    -> Chat Completions response
```

The mock processor is **not** the real `llm-d` Go EPP. The tests prove the
transport and routing integration, not `llm-d` scheduling policy.

## Body-based routing and request policies

Set `defer_request_processing: true` on the `ext_proc` filter when EPP
needs headers or request data produced by gateway policies. The option
requires `processing_mode.request_body_mode: full_duplex_streamed`.
Place `model_to_header`, authorization, quota, guardrails, and header
policies before `ext_proc`. Place `endpoint_selector` after it.
The [llm-d routing example](../../examples/configs/llmd-ext-proc-routing.yaml)
strips client-supplied base-model, objective, and model-rewrite headers.
Configure a trusted request-header policy after that cleanup to supply the
values permitted for the authorized route.

Praxis reads the request body under its configured body limits. The normal
request phase sends the effective headers and the buffered body to EPP
before selecting a worker. The picker receives only the inference headers in
the table, HTTP request metadata (`Host`, `Content-Type`, `Content-Length`),
and correlation headers already present after policy (`X-Request-ID`,
`traceparent`, `tracestate`). When present, `Content-Length` is calculated
from the body sent to EPP. Pending upstream header injections, including
credentials and IDs injected by `trace_context`, stay on the worker request.
Praxis clears any existing `x-gateway-destination-endpoint` before asking
EPP to supply the destination for `endpoint_selector`. EPP body replacements
are applied after endpoint selection, before forwarding. Chat response
bodies can still stream.

| Inference header | Who sets it and what reaches EPP |
|------------------|-----------------------------------|
| `X-Gateway-Model-Name` | Gateway-owned. Configure `model_to_header` to derive it from the request body and replace any incoming value before policy checks. |
| `X-Gateway-Base-Model-Name` | Optional for adapters. The model or adapter policy supplies the trusted base model when that capability is enabled. EPP receives the effective value when set. |
| `x-llm-d-inference-objective` (`x-gateway-inference-objective` on older routers) | Gateway policy can name an `InferenceObjective` for the request. |
| `x-llm-d-model-name-rewrite` (`x-gateway-model-name-rewrite` on older routers) | Gateway policy can supply the model name the authorized route uses on the worker. |
| `x-gateway-destination-endpoint` | Reserved for the gateway and EPP. The outbound picker request omits it. Praxis uses the endpoint returned by EPP. |

Use the `x-llm-d-*` spellings with llm-d Router v0.9.0 or later. Praxis
forwards both spellings when set, and the router prefers the canonical spelling
if both are present. See the [llm-d header reference](https://llm-d.ai/docs/0.10/api-reference/epp-http-headers#deprecated-aliases).

The option defaults to `false`, which keeps the incremental request-body
exchange for existing configurations. Use deferred mode when the first EPP
message needs the inference headers listed above.

In deferred mode, the EPP allowlist omits the optional fairness header
(`x-llm-d-inference-fairness-id`), the per-request latency SLO headers
(`x-llm-d-slo-ttft-ms` and `x-llm-d-slo-tpot-ms`), and their legacy aliases.
For ordinary request headers, deferred mode applies an EPP reply action of
`AppendIfExistsOrAdd` as a header set in the request phase. The reserved
destination header follows the selector handoff described above.

## Support boundary

- AI-owned, `publish = false`, `llm-d`-only. It is not general-purpose Envoy
  `ext_proc` support, and nothing is promised for non-`llm-d` uses.
- Prefer native Praxis filters for general extension work. Other use cases
  need a maintainer discussion on GitHub.
- The Grid demo should consume AI with `llmd-ext-proc` enabled for
  provider-gateway/`llm-d` testing, not depend on `praxis-proxy-ext-proc` or
  Praxis PR #780.

## Running the tests

Requires Docker or Podman for the `llm-d-inference-sim` container (pinned to
`v0.10.0`).

```console
cargo test -p praxis-tests-environment \
    --features llmd-ext-proc \
    -- --test-threads=1
```

The six environment tests:

| Test | Proves |
|------|--------|
| `simulator_chat_completion_routes_through_praxis` | Full request path through `ext_proc` + `endpoint_selector` to simulator |
| `simulator_spoofed_destination_header_ignored` | Client-supplied routing header cannot override processor decision |
| `simulator_repeated_requests_no_crosstalk` | Independent `ext_proc` streams per request |
| `simulator_health_endpoint_reachable` | Simulator `/health` returns 200 |
| `simulator_metrics_endpoint_reachable` | Simulator `/metrics` returns 200 with content |
| `simulator_processor_failure_returns_status_on_error` | Unreachable processor returns 503, no bypass |

## Dependency and build notes

The AI workspace tracks Praxis core `main` (pinned in `Cargo.lock`), which
provides the support APIs the `llm-d` `ext_proc` layer needs.
