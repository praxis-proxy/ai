# Experimental managed Responses and tools

Choose the owner of Responses state first. [Native forwarding](../quickstart.md)
needs no managed Responses features and leaves state and hosted tools to the
upstream. Adding `openai_responses_proxy` alone does not provide a complete
managed service. Gateway-managed validation, history, persistence, and tool
execution require an explicitly composed experimental pipeline.

## Prerequisites and build

Follow the [recipe prerequisites](README.md). For this local lab you need:

- A native `/v1/responses` inference backend on `127.0.0.1:3001` that supports
  function tools, returns complete Responses objects, and implements Responses
  SSE if you test streaming. A Chat-only backend needs a different adapter chain.
- A reachable MCP server on `127.0.0.1:8001/mcp`, with a harmless tool you are
  authorized to call, and a key if that server requires one.
- A writable working directory for SQLite. Use a fresh scratch directory for
  testing; history and tool output may contain sensitive content.

Build all experimental OpenAI groups plus SQLite explicitly:

```console
cargo build -p praxis-ai-proxy --features openai-all,store-sqlite
export BACKEND_API_KEY='<inference-backend-key>'
```

This feature selection exceeds the default image. Do not assume the published
standard image contains these filters. The [feature reference](../features.md#cargo-features)
explains narrower selections and PostgreSQL alternatives.

## Complete local pipeline

Save as `managed-responses.yaml`. This is a single-owner loopback lab, with
MCP discovery, bounded model/tool looping, logical SSE composition, local
Conversations, and SQLite persistence. It has no public ingress authentication.
Do not expose it as a multi-tenant service without authenticated owner scoping.

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
        database_url: "sqlite://managed-lab.db?mode=rwc"
        responses_table: openai_responses
        conversations_table: openai_conversations
      - filter: openai_conversations
        backend: sqlite
        database_url: "sqlite://managed-lab.db?mode=rwc"
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

The private-address opt-ins permit the local backend and MCP service. Production
callouts need destination restrictions, TLS, authentication, and an outbound
policy appropriate to each service. The inference key is injected only into
the inference cluster. Supply MCP authentication through that tool entry's
`authorization` field when needed; never put an inference bearer or a static
catch-all secret into an outbound chain for arbitrary MCP URLs.

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

Expect a completed Responses object containing an ID and output items. Save
that ID if you want to continue with `previous_response_id`. To test SSE, add
`"stream":true` and `curl -N`; expect one logical Responses lifecycle ending
in `response.completed` on success, or an explicit terminal failure. A native
backend must supply the corresponding wire contract. A valid YAML file does
not prove any of those runtime behaviors.

To request a harmless MCP tool, replace the model, tool name, and prompt with
values from your server, then send:

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

If the model selects that tool, expect an `mcp_approval_request` instead of
execution. Resume with the returned response ID, the same tools declaration,
and an input item `{"type":"mcp_approval_response","approval_request_id":"<returned-call-id>","approve":true}`.
For example, approve with this complete request after replacing placeholders:

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

After approval, expect one authorized tool call, a model continuation, and final
output. A denial must execute zero tool calls. Verify MCP recordings and
inference round counts; output text alone does not prove execution. The
[existing loop example](../../examples/configs/openai/responses/agentic-loop.yaml)
contains the full approval/denial lifecycle and additional dispatchers.

Client function tools are returned to the client for execution. Recognizing a
tool declaration does not imply a server executor exists. Managed
`code_interpreter`, `background:true`, and non-null reusable `prompt`
references are rejected; direct provider-owned passthrough is a different path.
See [managed validation](../filters/openai_responses_request.md),
[MCP dispatch](../filters/openai_mcp_dispatch.md), and
[retained-payload bounds](../architecture/responses-retained-payload.md).

## Conversations and state

With this pipeline running, local Conversations CRUD needs no model call:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/conversations \
  -H 'content-type: application/json' -d '{}'
```

Expect a Conversation ID. Use it in a subsequent Responses request's
`conversation` field; completed input/output items append to the owner-scoped
Conversation. The store and Conversations filters share the database and run
before rehydration/inference; reverse response order appends the conversation
turn before storing its successful response. See the
[exact Conversations contract](../filters/openai_conversations.md) and
[store lifecycle](../architecture/response-store.md). In the unified example,
an OpenAI-selected route forwards directly and the provider owns history.

For Chat-only inference, use the complete
[Responses-to-Chat configuration](../../examples/configs/openai/responses/responses-to-chat-completions.yaml)
with this build and a Chat backend at port 3001. It provides translation and
state, not the model/tool loop above. Read its
[compatibility gaps](../conformance/translation-matrix.md#known-gaps), especially
continuation history and operations without a Chat equivalent. Do not claim
WebSocket support from an HTTP POST translation recipe.

## Expand deliberately

The [unified full-flow example](../../examples/configs/openai/responses/full-flow-agentic.yaml)
adds provider-owned passthrough, managed providers, hosted web/file search,
files, compaction, and selected MCP connectors. It also needs the configured
service endpoints, secrets, and trusted ingress assertions; it is not a
one-service quickstart. Files API routing does not create a file service;
file resolution needs a reachable Files API, and file-search dispatch needs a
vector-store service. Compaction requires a summarization backend. MCP
classification/brokering and A2A routing are separate protocol capabilities,
described in [agentic protocols](../architecture/agentic-protocols.md).

## Cleanup

Stop the proxy before removing lab state. Stop the backend and MCP server only
if started for this recipe. Unset `BACKEND_API_KEY` in the proxy shell and
`MODEL` in the client shell. Delete your local `managed-responses.yaml` and,
only if this was a disposable lab, `managed-lab.db` and its `-wal`/`-shm`
sidecars. Do not remove a shared or production store. Follow
[store migrations](../store/schema-migration.md) for deployed state.
