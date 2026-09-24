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

### Committed example that translates to Chat Completions

Not every backend serves the Anthropic Messages API natively. When the backend
speaks only OpenAI Chat Completions,
[`examples/configs/anthropic/messages-to-openai-vllm.yaml`](../examples/configs/anthropic/messages-to-openai-vllm.yaml)
is the counterpart of the native config above: it keeps the identical
three-boundary credential isolation but adds the
`anthropic_messages_to_chat_completions[_stream]` translation filters and a
`path_rewrite` that maps `POST /v1/messages` to `/v1/chat/completions`. The
client still speaks the Anthropic wire format; Praxis rewrites both the request
and the response, so vLLM only ever sees OpenAI Chat Completions.

`/v1/messages/count_tokens` has no Chat Completions equivalent, so the
`path_rewrite` is anchored to `^/v1/messages$` and leaves it unrewritten. A
Chat-Completions-only backend returns 404 for it, and Claude Code degrades
gracefully because it treats native token counting as best-effort.

### Acceptance test: real Claude Code drives vLLM, native and transformed (issue #1025)

[`tests/integration/tests/suite/claude_code_vllm.rs`](../tests/integration/tests/suite/claude_code_vllm.rs)
proves the full flow end to end on both paths: a pinned real Claude Code
executable completes a deterministic multi-step coding task through Praxis
against a real vLLM backend with the native passthrough and translation configs.
Each path runs under two permission scenarios, producing four independent GPU
acceptance cases:

- `acceptEdits`, with the task's Read, Edit, and Bash tools preapproved, is the
  deterministic harness baseline.
- `auto`, with no tools preapproved and `CLAUDE_CODE_AUTO_MODE_SERVER=0`, forces
  Claude Code to initiate the classifier model requests through Praxis before
  it can Edit the file or run Bash.

```text
native      Claude Code ─► Praxis (messages-native-vllm.yaml)    ─► vLLM /v1/messages
transformed Claude Code ─► Praxis (messages-to-openai-vllm.yaml) ─► vLLM /v1/chat/completions
```

Each scenario asserts only what a live run uniquely proves: the client completes
the task through Praxis against a real backend — a non-timed-out, successful
exit, the exact uppercase-derived output file, a harness-owned verification
marker written only when the task's `verify.sh` confirms the compare, and a
non-empty final summary in the client's stream-json output. Wire fidelity is
proven deterministically against controlled fake backends, not observed in the
live run:

- Native passthrough (no `chat/completions` reshaping, the exact served model on
  every inference body), credential isolation, and native token counting in
  [`tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs`](../tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs).
- The translation path — `/v1/messages` rewritten to `/v1/chat/completions`, the
  Anthropic request body translated (system hoisted into a Chat Completions
  message) and the OpenAI response translated back into an Anthropic message, and
  the same credential isolation — in
  [`tests/integration/tests/suite/examples/anthropic_messages_to_openai_vllm.rs`](../tests/integration/tests/suite/examples/anthropic_messages_to_openai_vllm.rs).

Both cases share the same credential isolation guarantee: the client's native
`x-api-key` and its gateway `Authorization: Basic` credential are both stripped
so only the injected backend bearer reaches vLLM, and an unauthenticated caller
is rejected by `basic_auth`.

The tests are gated on live infrastructure and skip unless every required
variable is set. Run them locally against your own pinned binary and backend
(one vLLM container can serve both surfaces):

```console
PRAXIS_TEST_CLAUDE_CODE_BIN=/absolute/path/to/claude \
PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
PRAXIS_TEST_VLLM_MODEL=<exact-served-model-name> \
VLLM_API_KEY=<backend-bearer-token> \
  cargo test -p praxis-tests-integration --test suite \
  claude_code_vllm::pinned_claude_code_drives -- --nocapture
```

Set `PRAXIS_TEST_CLAUDE_CODE_NETNS` (and `PRAXIS_TEST_LISTEN_ADDRESS` to the
host-side veth address, which Praxis then binds) to launch the client inside a
restricted Linux network namespace that can reach only Praxis, proving it cannot
bypass the proxy.

**Auto-mode classifier limitation.** Praxis and vLLM do not implement
Anthropic's server-side auto-mode classifier protocol (`safeguards` and
`safeguard_results`). Do not enable that mode with
`CLAUDE_CODE_AUTO_MODE_SERVER=1`, and do not rely on Claude Code's server-side
default when using this backend. Set `CLAUDE_CODE_AUTO_MODE_SERVER=0` so Claude
Code sends its additional classifier inference requests through Praxis. In
Anthropic's terminology this is the "local" classifier path, but classification
still uses model requests; it is not an on-device classifier. See
[Auto mode classifier billing](https://code.claude.com/docs/en/auto-mode-classifier-billing)
for the client behavior and billing distinction.

**Pins and scheduled acceptance.** The pinned Claude Code version and launch
setup, the vLLM image digest, served model, and startup request matrix live in
[`tests/integration/fixtures/claude-code-cli/pin.toml`](../tests/integration/fixtures/claude-code-cli/pin.toml).
The `vllm-gpu-claude-acceptance` job in
[`.github/workflows/vllm-integration.yaml`](../.github/workflows/vllm-integration.yaml)
runs all four scenarios sequentially against one shared vLLM container, each
exactly once (no retry), on every nightly GPU run. It can also run independently
through the `run_claude_acceptance` workflow-dispatch input. Runtime pins must be
complete; the job fails fast with an explanatory error if any required value is
empty or still contains `TBD`.

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
- Reports a matched stop sequence as `stop_reason:
  stop_sequence` with the matched value when the backend
  names it in vLLM's choice-level `stop_reason`; a
  backend that only returns `finish_reason: stop` (for
  example the OpenAI API) cannot distinguish a stop
  sequence from a natural stop, so the response reports
  `end_turn`
- Maps `metadata.user_id` to `safety_identifier` as its
  SHA-256 hex digest, `output_config.effort` to
  `reasoning_effort`, and a `json_schema`
  `output_config.format` (or the deprecated
  `output_format`) to a strict `response_format`; any
  other `output_config` key is rejected with a 400
- Rejects `service_tier`, `container`, `inference_geo`
  and `mcp_servers` with a 400, because the translated
  response cannot report their effect truthfully; the
  same applies to Chat Completions fields whose output
  the translated response would discard (`n`,
  `logprobs`, `top_logprobs`, `audio`, `modalities`,
  `functions`, `function_call`, `web_search_options`,
  `moderation`); a `null` or the field's documented
  default (for example `n: 1`) is dropped instead
- Drops `thinking` and `context_management` with a log
  warning; Claude Code sends both on every request and
  Chat Completions has no equivalent
- Forwards every other field untouched (for example
  `top_k`) and leaves its validation to the backend
- Drops `thinking` content blocks with a log warning
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
