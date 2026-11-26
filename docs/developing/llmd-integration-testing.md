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
