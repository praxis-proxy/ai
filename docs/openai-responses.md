# OpenAI Responses API

Praxis AI supports the OpenAI Responses API (`/v1/responses`)
and related endpoints through composable filters on the
[AI Gateway][overview]. On the gateway data plane, Responses
traffic is **routed** (body classification and cluster
selection), **managed** (validation and policy), **enriched**
(stored responses, conversation history, rehydration, and
request rebuilding from pipeline state), and **parsed** (SSE
events, tools, token usage). Operators can run minimal
passthrough to OpenAI or a full stateful pipeline with local
persistence and multi-turn context.

This guide shows how to wire those filters with runnable
examples and the outcomes to expect. For OpenAI request and
response schemas, see the
[OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses).

[overview]: overview.md

## Filters

Deploy filters in pipeline order: classify first, then validate,
enrich/store/rehydrate as needed, route upstream.

| Filter | Phase | Purpose |
| ------ | ----- | ------- |
| `openai_conversations` | Request/response | Local `/v1/conversations` CRUD |
| `openai_responses_format` | Request | Classify format; promote routing headers |
| `openai_responses_validate` | Request | Parameter checks; generate IDs |
| `openai_tool_parse` | Request | Parse tools for branch routing |
| `openai_responses_rehydrate` | Request | Load history from `previous_response_id` |
| `openai_responses_proxy` | Request | Rebuild body from `ResponsesState` |
| `openai_response_store` | Request/response | Persist responses; local GET/DELETE |
| `openai_stream_events` | Request/response | Accumulate streaming SSE events |
| `openai_responses_model_rewrite` | Request body | Rewrite `model` field |

Per-filter configuration: [filter reference](filters/reference.md).

## Run an example locally

Example configs under `examples/configs/openai/responses/` use
local upstream ports (commonly `127.0.0.1:8000`). Start a stub
or inference backend on those ports before sending traffic.

```console
cargo run -p praxis-ai-proxy -- \
  -c examples/configs/openai/responses/request-validate.yaml
```

In another terminal, send requests to `http://127.0.0.1:8080`.
See [examples/README.md](../examples/README.md#openai) for the
full catalog.

## Examples

Each example below uses a config under
`examples/configs/openai/`. Start the proxy with
`-c <path>` and a stub backend on the configured upstream
port (see [Run an example locally](#run-an-example-locally)).
Integration tests live under
`tests/integration/tests/suite/examples/`.

### Validate and forward

[request-validate.yaml](../examples/configs/openai/responses/request-validate.yaml)
- `openai_responses_format` then `openai_responses_validate`
before routing. Smallest **manage**-plane demo.

Valid request:

```console
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1","input":"Hello, world!"}'
```

**Expect:** HTTP **200** and the upstream body unchanged
(tests use an echo backend returning `ok`).

Invalid request:

```console
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1","input":"test","stream":true,"background":true}'
```

**Expect:** HTTP **400** (SSE `type: error` when `stream: true`;
JSON error otherwise). Upstream is not called.

**Tests:** `openai_responses_validate.rs`

### Route by body format

[format-routing.yaml](../examples/configs/openai/responses/format-routing.yaml)
- **route** plane: `openai_responses_format` promotes
`x-praxis-ai-format`; router picks the cluster.

```console
# Responses API shape -> responses-backend cluster
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","input":"Hello"}'
```

**Expect:** HTTP **200**, response body `responses-backend`.

```console
# Chat Completions shape -> chat-backend cluster
curl -sS -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"Hi"}]}'
```

**Expect:** HTTP **200**, response body `chat-backend`.

**Tests:** `openai_responses_format.rs`

### Enrichment: persist and serve locally

[response-store.yaml](../examples/configs/openai/responses/response-store.yaml)
- **enrich** plane: `openai_response_store` persists
non-streaming POST responses and serves GET/DELETE at the
gateway without calling upstream.

```console
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1","input":"Hello"}'
```

**Expect:** HTTP **200** and the upstream Responses JSON
returned to the client (e.g. `id`, `output`, ...). The same
response is written to the configured store (SQLite or
PostgreSQL).

```console
curl -sS http://127.0.0.1:8080/v1/responses/resp_abc
```

**Expect:** HTTP **200** and the stored JSON from the
gateway - not a round trip to the inference backend.

```console
curl -sS -X DELETE http://127.0.0.1:8080/v1/responses/resp_abc
```

**Expect:** HTTP **200** and `{"id":"resp_abc","deleted":true}`.

**Tests:** `openai_response_store.rs`

### Enrichment: rehydrate a previous turn

[rehydrate.yaml](../examples/configs/openai/responses/rehydrate.yaml)
- **enrich** plane: validates `previous_response_id`
against the store and loads history into pipeline state
before forwarding.

Turn 1 - seed the store:

```console
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1","input":"Hello"}'
```

**Expect:** HTTP **200** and upstream Responses JSON (e.g.
`id: resp_first`, `status: completed`).

Turn 2 - follow-up with stored ID:

```console
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1","input":"What next?","previous_response_id":"resp_first"}'
```

**Expect:** HTTP **200**. Rehydration succeeds when the
stored response exists and is `completed`; the request body
reaches upstream unchanged (`input` and
`previous_response_id` preserved). A missing or incomplete
stored ID is rejected by the rehydrate filter.

**Tests:** `rehydrate.rs`

### Example index

| Goal | Example | Integration tests |
| ---- | ------- | ----------------- |
| Validate parameter combinations | [request-validate.yaml](../examples/configs/openai/responses/request-validate.yaml) | `openai_responses_validate.rs` |
| Route by detected body format | [format-routing.yaml](../examples/configs/openai/responses/format-routing.yaml) | `openai_responses_format.rs` |
| Route stateful vs stateless mode | [responses-routing.yaml](../examples/configs/openai/responses/responses-routing.yaml) | `responses_routing.rs` |
| Persist responses; local GET/DELETE | [response-store.yaml](../examples/configs/openai/responses/response-store.yaml) | `openai_response_store.rs` |
| Rehydrate `previous_response_id` | [rehydrate.yaml](../examples/configs/openai/responses/rehydrate.yaml) | `rehydrate.rs` |
| Tool-based branch routing | [tool-routing.yaml](../examples/configs/openai/responses/tool-routing.yaml) | `openai_tool_parse.rs` |
| Rewrite model before upstream | [model-rewrite.yaml](../examples/configs/openai/responses/model-rewrite.yaml) | `openai_responses_model_rewrite.rs` |
| Accumulate streaming SSE state | [stream-events.yaml](../examples/configs/openai/responses/stream-events.yaml) | `openai_stream_events.rs` |
| Local conversations CRUD | [conversations.yaml](../examples/configs/openai/conversations/conversations.yaml) | `openai_conversations.rs` |
| Full multi-filter pipeline | [full-flow.yaml](../examples/configs/openai/responses/full-flow.yaml) | `session_replay.rs` |

## Production passthrough to OpenAI

To forward validated traffic to `api.openai.com`, add TLS upstream
config and supply a client or injected API key. A minimal chain:

```yaml
filter_chains:
  - name: responses
    filters:
      - filter: openai_responses_format
      - filter: openai_responses_validate
      - filter: router
        routes:
          - path_prefix: "/v1"
            cluster: openai
      - filter: load_balancer
        clusters:
          - name: openai
            endpoints: ["api.openai.com:443"]
            tls:
              sni: "api.openai.com"
```

```console
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-4o","input":"Hello"}'
```

**Expect:** HTTP **200** and an OpenAI Responses JSON body (`id`,
`output`, `usage`, ...) - same shape as calling the provider
directly. Classification and validation are transparent on the
happy path. For credential injection instead of client-supplied
keys, see [credential-injection.yaml](../examples/configs/credential-injection.yaml).

## Stateful pipeline

Multi-turn traffic with storage, rehydration, and tool routing
uses more filters. Reference implementation:
[full-flow.yaml](../examples/configs/openai/responses/full-flow.yaml).

Typical order:

```text
openai_conversations -> openai_responses_format -> openai_responses_validate
  -> openai_tool_parse -> openai_response_store -> openai_stream_events
  -> openai_responses_rehydrate -> openai_responses_proxy -> router -> load_balancer
```

A request is **stateful** when any of these hold:
`previous_response_id`, non-empty `tools`, `store: true`
(default), `background: true`, `conversation`, or `prompt.id`.

**What to look for:** persisted `resp_*` IDs, local
`GET /v1/responses/{id}`, conversation append-back, and replay
coverage in `tests/integration/fixtures/replay/codex/responses-basic.json`.

Storage design: [Response store](architecture/response-store.md).

## Related

- [AI inference](architecture/ai-inference.md)
- [Token counting](token-counting.md)
- [Filter reference](filters/reference.md)
- [Example configs](../examples/README.md#openai)
