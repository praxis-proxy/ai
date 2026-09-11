# Anthropic Messages API

Praxis supports the Anthropic Messages API
(`/v1/messages`) through five composable filters.
Operators can route, validate JSON envelopes, and transform
Anthropic requests to reach any backend.

## Filters

| Filter | Purpose |
| ------ | ------- |
| `anthropic_messages_format` | Classify requests and promote routing facts to headers |
| `anthropic_validate` | Validate the JSON request envelope before forwarding |
| `anthropic_messages_protocol` | Header management for native `/v1/messages` backends |
| `anthropic_messages_to_chat_completions` | Bidirectional body transformation to the Chat Completions wire shape |
| `anthropic_messages_to_chat_completions_stream` | SSE event transformation (per-chunk streaming, conformant with Inference Proxy Conformance Guidelines) |

## Passthrough to vLLM

Route Anthropic requests directly to a backend that
supports `/v1/messages` natively (e.g. vLLM with
Anthropic endpoint enabled).

```yaml
listeners:
  - name: gateway
    address: "0.0.0.0:8080"
    filter_chains: [anthropic]

filter_chains:
  - name: anthropic
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue

      - filter: anthropic_validate

      - filter: anthropic_messages_protocol
        default_version: "2023-06-01"

      - filter: router
        routes:
          - path_prefix: "/"
            cluster: vllm

      - filter: load_balancer
        clusters:
          - name: vllm
            endpoints:
              - "127.0.0.1:8000"
```

Test:

```console
curl http://localhost:8080/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "max_tokens": 100,
    "system": "Reply concisely.",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### Committed example with credential isolation

[`examples/configs/anthropic/messages-native-vllm.yaml`](../examples/configs/anthropic/messages-native-vllm.yaml)
is the hardened version of the config above and the one a real Claude Code
client uses to reach a native vLLM backend. It deliberately omits the
`anthropic_messages_to_chat_completions[_stream]` translation filters — vLLM
speaks `/v1/messages` natively, so translation would be lossy — and isolates
three distinct credentials, each at its own boundary:

- `basic_auth` authenticates the *client* to the gateway. A trusted caller
  (e.g. Claude Code via `ANTHROPIC_CUSTOM_HEADERS`) presents an
  `Authorization: Basic ...` gateway credential; the filter verifies it and,
  with `strip_authorization: true`, removes it so it never reaches vLLM. The
  password resolves from `GATEWAY_AUTH_PASSWORD` at pipeline build time.
- `headers` removes the client's native `x-api-key` so a caller's Anthropic
  key never reaches vLLM.
- `credential_injection` adds the backend's own `Authorization: Bearer`
  token from the `VLLM_API_KEY` environment variable, resolved at pipeline
  build time.

Together these guarantee only authenticated callers are served, the client
cannot smuggle either of its own credentials to the backend, and vLLM only ever
sees the server-owned bearer token. The `router` forwards `/v1/messages`,
`/v1/messages/count_tokens`, and `/` unchanged, so token counting and model
discovery stay native too.

### Acceptance test: real Claude Code drives native vLLM (issue #1025)

[`tests/integration/tests/suite/claude_code_vllm.rs`](../tests/integration/tests/suite/claude_code_vllm.rs)
proves the full flow end to end: a pinned real Claude Code executable completes
a deterministic multi-step coding task while Praxis routes native Anthropic
Messages traffic straight to a vLLM backend that serves the Anthropic Messages
API natively.

```text
Claude Code ─► Praxis (messages-native-vllm.yaml) ─► vLLM
```

The test asserts only what a live run uniquely proves: the client completes the
task through Praxis against a real backend — a non-timed-out, successful exit,
the exact uppercase-derived output file, a harness-owned verification marker
written only when the task's `verify.sh` confirms the compare, and a non-empty
final summary in the client's stream-json output. Wire fidelity — native
passthrough (no `chat/completions` reshaping, the exact served model on every
inference body), credential isolation (the client's native `x-api-key` and its
gateway `Authorization: Basic` credential are both stripped so only the injected
backend bearer reaches vLLM, and an unauthenticated caller is rejected by
`basic_auth`), and native token counting — is proven deterministically against
controlled fake backends in
[`tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs`](../tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs),
not observed in the live run.

The test is gated on live infrastructure and skips unless every required
variable is set. Run it locally against your own pinned binary and backend:

```console
PRAXIS_TEST_CLAUDE_CODE_BIN=/absolute/path/to/claude \
PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
PRAXIS_TEST_VLLM_MODEL=<exact-served-model-name> \
VLLM_API_KEY=<backend-bearer-token> \
  cargo test -p praxis-tests-integration --test suite \
  claude_code_vllm::pinned_claude_code_drives_native_vllm_through_full_flow -- --exact
```

Set `PRAXIS_TEST_CLAUDE_CODE_NETNS` (and `PRAXIS_TEST_LISTEN_ADDRESS` to the
host-side veth address, which Praxis then binds) to launch the client inside a
restricted Linux network namespace that can reach only Praxis, proving it cannot
bypass the proxy.

**Pins and qualification.** The pinned Claude Code version and launch flags, the
vLLM image digest, served model, and startup request matrix live in
[`tests/integration/fixtures/claude-code-cli/pin.toml`](../tests/integration/fixtures/claude-code-cli/pin.toml).
The `claude-code-native-vllm` job in
[`.github/workflows/vllm-integration.yaml`](../.github/workflows/vllm-integration.yaml)
runs the test exactly once (no retry) on `workflow_dispatch` only, so PR and
merge-queue CI stay green while qualification is pending. Before it can run, a
model must pass the consecutive qualification runs recorded in the manifest, the
`TBD-at-qualification` pins (served model, image digest, revision, Claude Code
archive url + sha256) must be filled in, and the manifest `status` set to
`qualified`; the job also requires the `VLLM_API_KEY` repository secret. Until
then it fails fast with an explanatory error rather than running against
unqualified pins.

## Passthrough to Anthropic API

Route to `api.anthropic.com` with credential
injection for the `x-api-key` header.

```yaml
listeners:
  - name: gateway
    address: "0.0.0.0:8080"
    filter_chains: [anthropic]

filter_chains:
  - name: anthropic
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue

      - filter: anthropic_validate

      - filter: anthropic_messages_protocol
        default_version: "2023-06-01"

      - filter: headers
        request_set:
          - name: Host
            value: "api.anthropic.com"

      - filter: router
        routes:
          - path_prefix: "/"
            cluster: anthropic

      - filter: credential_injection
        clusters:
          - name: anthropic
            header: x-api-key
            env_var: ANTHROPIC_API_KEY
            strip_client_credential: true

      - filter: load_balancer
        clusters:
          - name: anthropic
            tls:
              sni: "api.anthropic.com"
            endpoints:
              - "api.anthropic.com:443"
```

Test:

```console
curl http://localhost:8080/v1/messages \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-haiku-4-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Transformation to OpenAI Backend

Transform Anthropic requests to OpenAI Chat
Completions format for backends that only speak
OpenAI (e.g. llm-d with disaggregation, KServe
without Anthropic support).

```yaml
listeners:
  - name: gateway
    address: "0.0.0.0:8080"
    filter_chains: [transform]

filter_chains:
  - name: transform
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue

      - filter: anthropic_validate

      - filter: anthropic_messages_to_chat_completions
        max_body_bytes: 1048576

      - filter: anthropic_messages_to_chat_completions_stream
        response_conditions:
          - when:
              headers:
                content-type: "text/event-stream"

      - filter: path_rewrite
        replace:
          pattern: "^/v1/messages$"
          replacement: "/v1/chat/completions"
        conditions:
          - when:
              path_prefix: "/v1/messages"

      - filter: router
        routes:
          - path_prefix: "/"
            cluster: vllm

      - filter: load_balancer
        clusters:
          - name: vllm
            endpoints:
              - "127.0.0.1:8000"
```

The `anthropic_messages_to_chat_completions` filter:
- Hoists `system` to an OpenAI system message
- Flattens content blocks (text, image, tool_use,
  tool_result, document, search_result)
- Marks `tool_result.is_error` in translated tool
  message text because Chat Completions has no
  equivalent tool-result error flag
- Maps `stop_sequences` to `stop`,
  `tool_choice` semantics, tool definitions
- Preserves `top_k` as an extra body parameter
- Drops `thinking` blocks with a log warning
- Transforms the response back to Anthropic format
- Normalizes pre-stream upstream 4xx/5xx responses into
  Anthropic error envelopes for both streaming and
  non-streaming requests
- Preserves the upstream status and request ID while
  clearing stale representation headers on rewritten
  response bodies
- Preserves original `finish_reason` in filter
  metadata as `openai.finish_reason`

Add `anthropic_messages_to_chat_completions_stream` with a `text/event-stream`
response condition when the backend may return streaming
Chat Completions SSE. Keep it response-gated so normal
JSON responses stay on the buffered `anthropic_messages_to_chat_completions`
path.

## Filter Configuration Reference

### `anthropic_messages_format`

Classifies requests by body structure, then promotes
ambiguous `/v1/messages` or `anthropic-version`
requests to Anthropic Messages when the body otherwise
looks like Chat Completions.

```yaml
filter: anthropic_messages_format
on_invalid: continue      # continue | reject
max_body_bytes: 1048576    # 1 MiB
headers:
  format: x-praxis-ai-format
  model: x-praxis-ai-model
  stream: x-praxis-ai-stream
```

Body classification precedence:
1. `input` or object-valued `prompt` → OpenAI Responses
2. `messages` + `max_tokens` + Anthropic structural
   signals → Anthropic Messages
3. `messages` alone → OpenAI Chat Completions

`anthropic-version` and `/v1/messages` upgrade only the
ambiguous Chat Completions result to Anthropic Messages;
they do not override Responses-shaped bodies.

### `anthropic_validate`

Validates the proxy-owned JSON envelope before forwarding.
Backend-owned Anthropic semantics such as model availability,
message shape, role ordering, and token limits are deferred
to the backend.

```yaml
filter: anthropic_validate
max_body_bytes: 1048576    # 1 MiB
```

Checks: request body is present, valid JSON, and a JSON object.

### `anthropic_messages_protocol`

Injects `anthropic-version` header if absent.
No body transformation.

```yaml
filter: anthropic_messages_protocol
default_version: "2023-06-01"
```

### `anthropic_messages_to_chat_completions`

Bidirectional request/response transformation.
Non-streaming only; use `anthropic_messages_to_chat_completions_stream`
for SSE responses.

```yaml
filter: anthropic_messages_to_chat_completions
max_body_bytes: 1048576    # 1 MiB
```

### `anthropic_messages_to_chat_completions_stream`

Transforms Chat Completions SSE chunks to Anthropic SSE
events. Processes SSE chunks incrementally as they arrive.

```yaml
filter: anthropic_messages_to_chat_completions_stream
max_partial_event_bytes: 10485760
response_conditions:
  - when:
      headers:
        content-type: "text/event-stream"
```

## Running with Debug Logging

See filter activity in real time:

```console
RUST_LOG=debug cargo run -p praxis-ai-proxy -- -c config.yaml
```

Filter-specific logging:

```console
RUST_LOG=praxis_ai_apis=debug,praxis_ai_filters=debug cargo run -p praxis-ai-proxy -- -c config.yaml
```
