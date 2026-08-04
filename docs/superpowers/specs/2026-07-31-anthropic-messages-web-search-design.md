# Anthropic Messages Web Search Design

## Goal

Add a non-streaming web-search agentic-loop demo for the native Anthropic
Messages API. A client sends a Messages request to Praxis, Praxis forwards it
to a Messages-compatible vLLM backend, intercepts a model-generated
`WebSearch` tool call, executes the query through You.com, appends an Anthropic
`tool_result`, and asks the model for the final answer through Praxis core's
`iterative_request_router` (IRR).

The first runnable version uses a deterministic mock model backend adapted
from the existing Responses API web-search recording. A real vLLM deployment
can replace the mock by changing only the configured inference endpoint.

## Scope

This change provides:

- Native `/v1/messages` request and response bodies throughout the model loop.
- Non-streaming JSON model responses.
- You.com search through the existing bounded search-provider client.
- A Messages-specific web-search filter and request-scoped loop state.
- An IRR example configuration for a Messages-compatible vLLM backend.
- A deterministic, runnable mock model backend for local demonstration.
- Unit tests, a functional integration test, example documentation, and
  generated filter documentation.

This change does not provide:

- Incremental or buffered SSE support in the Messages agentic loop.
- A Claude Code end-to-end smoke test. Claude Code streams Messages requests,
  so that test belongs to the later Messages-over-IRR streaming work.
- General Anthropic client-tool execution. Non-web-search tool calls remain
  client-owned and are returned unchanged.
- Anthropic hosted server-tool wire types such as `server_tool_use` or
  `web_search_tool_result`.
- Parallel or mixed server/client tool dispatch in one model turn.
- vLLM installation or lifecycle management.

## Architecture

### Protocol-neutral search provider

Move the existing provider configuration, HTTP client, provider request
builders, result parsing, and result types out of the OpenAI Responses module
into an internal protocol-neutral web-search module under `apis/src/`.

The shared layer owns:

- Brave, Tavily, and You.com configuration.
- Secret-bearing API-key handling.
- `SubRequestClient` execution, deadlines, response-size limits, and failure
  modes.
- Provider-specific request construction and response parsing.
- The normalized `SearchResult` and `SearchOutcome` types.

It does not construct OpenAI or Anthropic error bodies and does not know either
protocol's tool-call representation. The existing `openai_web_search` filter
continues to produce Responses API output and errors through a thin protocol
adapter, preserving its public YAML contract and behavior.

Shared configuration validation accepts the owning filter name so diagnostics
remain `openai_web_search: ...` or `anthropic_web_search: ...` rather than
leaking the other protocol's name.

### `anthropic_web_search` filter

Add an `anthropic_web_search` filter under `apis/src/anthropic/web_search/` and
register it through `praxis_ai_filters::register_ai_filters` with the shared
server-level `SubRequestClient`.

The filter has the existing search-provider configuration surface:

```yaml
filter: anthropic_web_search
provider: you
api_key: ${WEB_SEARCH_API_KEY}
default_context_size: medium
timeout_ms: 10000
provider_failure_mode: closed
status_on_error: 502
max_body_bytes: 67108864
```

The filter uses bounded `StreamBuffer` request and response modes because IRR
must inspect a complete model response before deciding whether to return it or
run another step. It rejects an initial request containing `"stream": true`
with an Anthropic JSON `invalid_request_error` before making an upstream call.

### IRR-accounted Messages state

The filter keeps cross-iteration payloads in Praxis core's `IterationState`,
where `max_state_bytes` accounts for them. The original request remains in
`original_request.body`; after the first search, the latest serialized request
is stored under a namespaced `accumulator` key. The model response that caused
re-entry remains in `previous_response.body`.

The filter does not retain a parsed request, assistant content, query, or tool
identifier in a private request extension. On re-entry it parses the accounted
request and previous response into hook-local JSON values, then drops them after
serialization. The rebuilt `Bytes` allocation is shared between the accumulator,
`NextIterationBody`, and the active step body, so request propagation does not
copy the payload or bypass IRR accounting.

## Data Flow

### Initial request

1. `anthropic_messages_format` classifies the request.
2. `anthropic_validate` validates the proxy-owned Messages envelope.
3. `iterative_request_router` enters its `inference` step.
4. `anthropic_web_search` validates that the initial request is non-streaming;
   IRR already retains its body as `original_request`.
5. `anthropic_messages_protocol`, `router`, and `load_balancer` send the request
   to the Messages-compatible model backend.

### Tool-call response

The mock or real backend returns a normal Anthropic assistant message:

```json
{
  "id": "msg_search_1",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_search_1",
      "name": "WebSearch",
      "input": {"query": "potato"}
    }
  ],
  "stop_reason": "tool_use"
}
```

During the response-body phase, `anthropic_web_search` identifies a managed
call when all of the following are true:

- The response is a successful Anthropic message object.
- `stop_reason` is `tool_use`, or `end_turn` from a Messages-compatible vLLM
  response whose content still contains the validated tool-use block.
- Exactly one `tool_use` content block is present.
- Its name is exactly `WebSearch`.

The managed call must contain a non-empty string `input.query` no larger than
8 KiB in UTF-8; otherwise the filter returns the invalid-request error defined
below. A valid managed call
writes `anthropic_web_search.action = "loop"`; IRR then accounts for the full
response as `previous_response`. Any response with no managed call writes
`action = "done"` and is returned unchanged. This includes ordinary final text,
malformed or backend-owned error responses, non-web-search tool calls, multiple
tool calls, and mixed client/server calls.

### Search execution and request re-entry

On the next request-body phase, the filter reclassifies the accounted previous
response, moves out its complete assistant content, and executes the query
through the shared You.com client. It loads the latest request from the
accumulator (or the accounted original body on the first re-entry) and appends
two Messages turns:

```json
[
  {
    "role": "assistant",
    "content": [
      {
        "type": "tool_use",
        "id": "toolu_search_1",
        "name": "WebSearch",
        "input": {"query": "potato"}
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_search_1",
        "content": "[1] Potato - Wikipedia\nhttps://en.wikipedia.org/wiki/Potato\n..."
      }
    ]
  }
]
```

Search results use the existing bounded plain-text formatter for compatibility
with Messages-compatible vLLM backends. The filter resets `tool_choice` to
`{"type":"auto"}` on re-entry when that field is present, preventing a
client's first-turn forced tool choice from forcing repeated searches. It
serializes once, rejects with HTTP 413 if the rebuilt body exceeds the filter's
`max_body_bytes`, stores the bytes in the IRR accumulator, and installs the same
allocation as `NextIterationBody` and the active step body. It repairs
`Content-Type`; Praxis core repairs `Content-Length`.

The second backend response follows the same response decision. A final text
response is returned to the client. Another sole `WebSearch` call loops again
until IRR's configured `max_iterations` terminates the request.

## Error Handling

- `stream: true` returns HTTP 400 with an Anthropic
  `invalid_request_error` explaining that streaming is not supported by
  `anthropic_web_search`.
- A managed `WebSearch` call without a usable query returns HTTP 400 with an
  Anthropic `invalid_request_error`; it is not silently dispatched with an
  empty query.
- A managed query larger than 8192 UTF-8 bytes returns HTTP 400 before any
  provider callout.
- Provider failure in `closed` mode returns the configured status with an
  Anthropic JSON error body.
- Provider failure in `open` mode appends a successful `tool_result` whose
  content states that no search results were available, then continues the
  model loop.
- Invalid provider configuration fails pipeline construction.
- IRR owns overall deadlines, per-step deadlines, response/state byte limits,
  depth protection, and the infrastructure iteration cap.
- Non-2xx backend responses and client-owned tool calls pass through rather
  than being reinterpreted as managed searches or search failures.

## Example Configuration

Add `examples/configs/anthropic/messages-web-search.yaml` with this logical
pipeline:

```text
anthropic_messages_format -> anthropic_validate
  -> iterative_request_router(inference)
       anthropic_web_search -> anthropic_messages_protocol
       -> router -> load_balancer -> Messages backend
```

The IRR transition matches `anthropic_web_search.action = "loop"` and rejoins
the `inference` step; the default transition returns the current response. The
example uses `provider: you`, `${WEB_SEARCH_API_KEY}`, Praxis on
`127.0.0.1:8080`, and the model backend on `127.0.0.1:8000`.

## Deterministic Mock Backend

Add a runnable `praxis-test-utils` example that listens on
`127.0.0.1:8000` and implements the two-turn Messages behavior:

1. A request without the expected `tool_result` receives the `WebSearch`
   `tool_use` response.
2. A request containing the matching `tool_result` receives a final text
   response summarizing the potato fixture.

The response text and source facts are reformatted from
`tests/integration/fixtures/agentic_api/web_search_nonstreaming.json`, the
sanitized Responses/vLLM/You.com recording. The mock validates request shape
instead of advancing on connection count alone, so retries do not corrupt its
state.

The documented local demo is:

```console
cargo run -p praxis-test-utils --example anthropic_messages_web_search_mock
WEB_SEARCH_API_KEY=... cargo run -p praxis-ai-proxy -- \
  -c examples/configs/anthropic/messages-web-search.yaml
curl http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","max_tokens":1024,"stream":false,
       "messages":[{"role":"user","content":"Search for potato and summarize it."}],
       "tools":[{"name":"WebSearch","description":"Search the web",
                 "input_schema":{"type":"object","properties":{"query":{"type":"string"}},
                 "required":["query"]}}]}'
```

## Testing

### Unit tests

- Preserve all existing OpenAI web-search config and provider behavior after
  extracting the shared module.
- Validate a non-streaming initial Messages request without mutating its body.
- Reject `stream: true` before upstream execution.
- Detect exactly one valid `WebSearch` call and signal `loop`.
- Return final text and client-owned tool calls with `done`.
- Do not take ownership of multiple or mixed tool calls.
- Reject a managed call with a missing or empty query.
- Execute You.com search and construct the assistant plus user `tool_result`
  turns.
- Reset a present `tool_choice` to auto on re-entry.
- Map provider open/closed failures to the specified Messages behavior.
- Recover the complete assistant content from the accounted previous response.
- Read local provider-stub requests through their declared `Content-Length`.

### Functional integration test

Use a stateful capturing model backend and a local You.com-compatible search
stub. Assert that:

- The client receives the second model response.
- The model backend receives exactly two `/v1/messages` requests.
- The second model request preserves the original history and tool definition.
- The second model request contains the complete assistant `tool_use` turn.
- Its following user turn contains the matching `tool_result` and normalized
  search results.
- The search stub receives one You.com request with the expected query and
  API-key header.
- A `stream: true` request makes zero model and search calls.
- A large result that exceeds `max_body_bytes` returns 413 before model
  re-entry.
- A low `max_state_bytes` returns 413 once the rebuilt request would exceed
  IRR's retained-state cap, after one model call and one search call.

### Repository verification

Run focused crate and example tests, regenerate filter/example documentation,
run nightly formatting, and run the repository lint targets required for
changed Rust, examples, and generated docs.

## Follow-up

Messages-over-IRR streaming is a separate capability. It must define how
intermediate SSE events are suppressed, how tool-use deltas are accumulated,
how the final iteration is released incrementally, and how mid-stream provider
or iteration failures are represented. Once that exists, add the Claude Code
smoke demo using `ANTHROPIC_BASE_URL`.
