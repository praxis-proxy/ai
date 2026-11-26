# Experimental managed Responses and tools

Decide first who owns Responses state. [Native forwarding](../getting-started/build-from-source.md)
needs no managed Responses features and leaves state and hosted tools to the
upstream. For the gateway to validate requests, keep history, persist state,
and run tools, you compose the experimental pipeline below; adding
`openai_responses_proxy` on its own is not enough.

See the [Responses support and qualification table](../conformance/openai-responses-state-conformance.md#responses-support-and-qualification) for state parameters, SDK versions, and the native and translated vLLM test lanes.

## Prerequisites and build

Follow the [guide prerequisites](README.md). For this local lab you need:

- A native `/v1/responses` inference backend on `127.0.0.1:3001` that supports
  function tools, returns complete Responses objects, and implements Responses
  SSE if you test streaming. A Chat-only backend needs a different adapter
  chain (see [Conversations and state](#conversations-and-state)).
- An MCP server on `127.0.0.1:8001/mcp` with a harmless tool you may call, and
  its key if it requires one.
- A fresh scratch directory for SQLite, since history and tool output may hold
  sensitive content. The config below uses `/tmp/praxis-managed-lab`; create
  it with `mkdir /tmp/praxis-managed-lab`, which fails if an old lab is still
  there.

Build all experimental OpenAI groups plus SQLite:

```console
cargo build -p praxis-ai-proxy --features openai-all,store-sqlite
export BACKEND_API_KEY='<inference-backend-key>'
```

The published standard image does not contain these filters. The
[feature reference](../features.md#cargo-features) covers narrower selections
and PostgreSQL.

## Complete local pipeline

Save as `managed-responses.yaml`. It is a single-owner loopback lab with MCP
discovery, a bounded model/tool loop, logical SSE composition, local
Conversations, and SQLite persistence. It has no ingress authentication, so do
not expose it as a multi-tenant service without authenticated owner scoping.

```yaml
listeners:
  - name: managed
    address: "127.0.0.1:8080"
    filter_chains: [managed]
filter_chains:
  - name: managed
    filters:
      - filter: state_owner
        mode: single_tenant
        tenant_id: local-lab
      - filter: ai_operation
      - filter: openai_responses_request
        on_invalid: reject
      - filter: openai_tool_parse
      - filter: openai_responses_store
        backend: sqlite
        database_url: "sqlite:///tmp/praxis-managed-lab/managed-lab.db?mode=rwc"
        responses_table: openai_responses
        conversations_table: openai_conversations
      - filter: openai_conversations
        backend: sqlite
        database_url: "sqlite:///tmp/praxis-managed-lab/managed-lab.db?mode=rwc"
        conversations_table: openai_conversations
        items_table: openai_conversation_items
      - filter: openai_responses_rehydrate
      - filter: openai_mcp_tool_resolve
        timeout_ms: 5000
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 4
        max_response_bytes: 8388608
        max_stream_response_bytes: 67108864
        timeout_ms: 360000
        steps:
          - name: inference
            filters:
              - filter: openai_stream_events
              - filter: openai_mcp_dispatch
                outbound_chain:
                  name: mcp-outbound
                  filters:
                    - filter: request_id
                max_calls_per_round: 4
                max_parallel_calls: 2
                max_result_bytes: 1048576
                max_total_result_bytes: 4194304
              - filter: openai_agentic_loop
                max_infer_iters: 3
                max_retained_bytes: 67108864
              - filter: openai_responses_proxy
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: inference
              - filter: credential_injection
                clusters:
                  - name: inference
                    header: Authorization
                    env_var: BACKEND_API_KEY
                    header_prefix: "Bearer "
                    strip_client_credential: true
              - filter: load_balancer
                clusters:
                  - name: inference
                    read_timeout_ms: 300000
                    endpoints: ["127.0.0.1:3001"]
            on_result:
              - filter: openai_agentic_loop
                key: action
                value: loop
                next: inference
              - default: true
                done: true
insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
```

The private-address opt-ins allow the local backend and MCP server. In
production, each callout needs destination restrictions, TLS, authentication,
and its own outbound policy. The inference key goes only to the `inference`
cluster. Pass MCP credentials in the tool entry's `authorization` field; never
put the inference bearer or a catch-all static secret in an outbound chain that
reaches arbitrary MCP URLs.

## Validate and make a first request

```console
./target/debug/praxis-ai --validate -c managed-responses.yaml
./target/debug/praxis-ai -c managed-responses.yaml
```

In a client shell:

```console
export MODEL='<backend-model>'
curl --fail-with-body http://127.0.0.1:8080/v1/responses \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":\"Say hello\",\"store\":true}"
```

Expect a completed Responses object with an ID and output items. Keep the ID
to continue with `previous_response_id`. For SSE, add `"stream":true` and
`curl -N`; expect one logical Responses lifecycle that ends in
`response.completed`, or an explicit terminal failure. The backend must
implement the matching wire contract; passing `--validate` does not prove it.

To call the MCP tool, replace the model, tool name, and prompt with values from
your server:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "model":"<backend-model>",
    "input":"Use <safe-tool-name> to answer this harmless test question.",
    "tools":[{
      "type":"mcp",
      "server_label":"local-test",
      "server_url":"http://127.0.0.1:8001/mcp",
      "allowed_tools":["<safe-tool-name>"],
      "require_approval":"always"
    }]
  }'
```

If the model selects the tool, you get an `mcp_approval_request` instead of
execution. To approve, resume with the returned response ID, the same tools
declaration, and an `mcp_approval_response` input item:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "model":"<backend-model>",
    "previous_response_id":"<response-id-from-first-turn>",
    "tools":[{
      "type":"mcp",
      "server_label":"local-test",
      "server_url":"http://127.0.0.1:8001/mcp",
      "allowed_tools":["<safe-tool-name>"],
      "require_approval":"always"
    }],
    "input":[{
      "type":"mcp_approval_response",
      "approval_request_id":"<approval-request-id-from-first-turn>",
      "approve":true
    }]
  }'
```

After approval, expect one tool call, a model continuation, and final output.
A denial executes no tool calls. Check MCP server recordings and inference
round counts, because output text alone does not prove the tool ran. The
[loop example](../../examples/configs/openai/responses/agentic-loop.yaml) covers
the full approval and denial lifecycle and more dispatchers.

Client function tools go back to the client to execute; a recognized tool
declaration does not mean the gateway has an executor for it. The managed path
rejects `code_interpreter`, `background:true`, and non-null reusable `prompt`
references (direct provider-owned passthrough is a separate path). See
[managed validation](../filters/apis/openai_responses_request.md),
[MCP dispatch](../filters/agents/openai_mcp_dispatch.md), and
[retained-payload bounds](../architecture/responses-retained-payload.md).

## Conversations and state

Local Conversations CRUD on this pipeline needs no model call:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/conversations \
  -H 'content-type: application/json' -d '{}'
```

Pass the returned Conversation ID in a later Responses request's
`conversation` field; completed input and output items append to that
owner-scoped Conversation. The store and Conversations filters share one
database and run before rehydration and inference. In response order they run
in reverse, so the conversation turn is appended before its successful response
is stored. See the [Conversations contract](../filters/agents/openai_conversations.md)
and [store lifecycle](../architecture/response-store.md).

For a Chat-only backend on port 3001, use the complete
[Responses-to-Chat configuration](../../examples/configs/openai/responses/responses-to-chat-completions.yaml)
with this build. It provides translation and state but not the model/tool loop
above. Read its [compatibility gaps](../conformance/translation-matrix.md#known-gaps),
especially continuation history and operations with no Chat equivalent. This
HTTP POST recipe does not provide WebSocket support.

## Expand deliberately

The [unified full-flow example](../../examples/configs/openai/responses/full-flow-agentic.yaml)
adds provider-owned passthrough (an OpenAI-selected route forwards directly and
the provider owns history), managed providers, hosted web and file search,
files, compaction, and selected MCP connectors. Each needs its own service
endpoint, secrets, and trusted ingress assertions: file resolution needs a
reachable Files API, file-search dispatch needs a vector-store service, and
compaction needs a summarization backend. MCP classification and brokering and
A2A routing are separate capabilities, described in
[agentic protocols](../architecture/agentic-protocols.md).

## Cleanup

Stop the proxy before removing lab state, and stop the backend and MCP server
if you started them for this lab. Unset `BACKEND_API_KEY` in the proxy shell and
`MODEL` in the client shell. Delete `managed-responses.yaml` and, for a
disposable lab only, `/tmp/praxis-managed-lab` with the database and its
`-wal`/`-shm` sidecars. For a
deployed store, follow [store migrations](../operations/store/schema-migration.md)
instead.

## Filter reference

- [`ai_operation`](../filters/core/ai_operation.md)
- [`state_owner`](../filters/core/state_owner.md)
- [`openai_responses_request`](../filters/apis/openai_responses_request.md)
- [`openai_tool_parse`](../filters/apis/openai_tool_parse.md)
- [`openai_responses_rehydrate`](../filters/agents/openai_responses_rehydrate.md)
- [`openai_responses_proxy`](../filters/agents/openai_responses_proxy.md)
- [`openai_mcp_tool_resolve`](../filters/agents/openai_mcp_tool_resolve.md)
- [`openai_agentic_loop`](../filters/agents/openai_agentic_loop.md)
- [`openai_mcp_dispatch`](../filters/agents/openai_mcp_dispatch.md)
- [`openai_stream_events`](../filters/apis/openai_stream_events.md)
- [`openai_responses_store`](../filters/agents/openai_responses_store.md)
- [`openai_conversations`](../filters/agents/openai_conversations.md)

Core filters such as `router`, `headers`, `credential_injection`, and
`load_balancer` are documented in the [Praxis core filter reference](https://github.com/praxis-proxy/praxis/blob/main/docs/filters/reference.md).
