# Anthropic Messages API

Praxis AI can forward Anthropic Messages (`/v1/messages`) to a native
backend, or translate supported requests to a Chat Completions backend.
For build, credentials, startup, requests, and cleanup, start with the
[operator recipes](guides/anthropic.md). This page covers protocol behavior
and qualification. The optional Vertex Anthropic path is in the
[client/provider table](getting-started/choose-client-and-provider.md).

Translation covers a bounded subset of the API. Before choosing it over a
native backend, review the rejected fields, the lossy opt-ins, and the
[known gaps](conformance/translation-matrix.md#known-gaps).

## Filters

| Filter | Purpose |
| ------ | ------- |
| [`anthropic_messages_request`](filters/apis/anthropic_messages_request.md) | Process the create-message body once: validate the envelope and promote routing facts |
| [`anthropic_messages_protocol`](filters/apis/anthropic_messages_protocol.md) | Header management for native `/v1/messages` backends |
| [`anthropic_messages_to_chat_completions`](filters/apis/anthropic_messages_to_chat_completions.md) | Bidirectional body transformation to the Chat Completions wire shape |
| [`anthropic_messages_to_chat_completions_stream`](filters/apis/anthropic_messages_to_chat_completions_stream.md) | Incremental Chat Completions SSE to Messages events |

Each linked page holds the filter's configuration reference.

`anthropic_messages_request` processes only `POST /v1/messages`. The endpoint
decides the protocol: a Chat Completions-shaped body on `/v1/messages` is still
a create-message request, and an Anthropic-shaped body on another path is not.
A missing `anthropic-version` header does not change this. `on_invalid`
handles a missing, malformed, or non-object body: `continue` (the default)
forwards it for the backend to answer, and `reject` refuses it with an
Anthropic-compatible client error.

## Passthrough to vLLM

Route Anthropic requests directly to a backend that serves `/v1/messages`
natively, such as vLLM with the Anthropic endpoint enabled.

```yaml
listeners:
  - name: gateway
    address: "127.0.0.1:8080"
    filter_chains: [anthropic]

filter_chains:
  - name: anthropic
    filters:
      - filter: anthropic_messages_request
        on_invalid: continue

      - filter: router
        routes:
          - path_prefix: "/"
            cluster: vllm

      - filter: load_balancer
        clusters:
          - name: vllm
            http:
              application_provider: vllm
            endpoints:
              - "127.0.0.1:8000"

      # After load_balancer so it can refuse the MCP connector for the
      # selected vLLM cluster (see "MCP connector" below).
      - filter: anthropic_messages_protocol
        default_version: "2023-06-01"

insecure_options:
  allow_private_endpoints: true # local development backend
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

### MCP connector

Anthropic executes its server-side MCP connector (`mcp_servers` plus
`mcp_toolset` tools, beta `mcp-client-2025-11-20`) itself, so vLLM cannot
honor it. When the selected cluster's `application_provider` is anything
other than `anthropic`, `anthropic_messages_protocol` refuses the request with
a 400 `invalid_request_error`. It must run after `load_balancer` to see the
selected cluster. A cluster tagged `application_provider: anthropic`, or an
untagged cluster, forwards the connector unchanged. The Chat Completions
translation always refuses it.

To serve the connector on these backends, put the experimental
[`anthropic_mcp_connector`](filters/agents/anthropic_mcp_connector.md) filter
(cargo feature `anthropic-mcp-connector-filter`) first in the inference step
of an `iterative_request_router`. Praxis then runs the connector itself; the
filter page describes tool listing, `{server}__{tool}` naming, the
`server_url` match, and `allow_client_servers: true`. Beyond that page:

- The filter drops `mcp_servers` and the `mcp-client-*` beta token, so the
  refusal above no longer applies. Usage is summed over all rounds.
- The tools sent to the model are the ones the `mcp_toolset` enables through
  `default_config` and per-tool `configs`.
- Both the native Messages route and the Chat Completions translation work.
- With `stream: true`, text and tool arguments arrive as the model produces
  them. Each MCP call appears as an `mcp_tool_use` block whose result
  references the same ID. A mixed round returns the MCP results and leaves
  client tools for the caller. Rounds share the first message ID,
  consecutive block indices, and one terminal usage delta and stop.
- A complete model `message_stop` lets Praxis start the next round or finish
  the response even if the backend keeps its HTTP body open.
- MCP and web search share the Messages SSE parser. It recognizes CR line
  endings immediately, rejects deltas for unopened blocks and malformed
  lifecycle fields, preserves citations and compaction data for replay, and
  carries provider metadata into the terminal delta.
- Streamed MCP results follow the round's model blocks. Buffered responses
  place each result right after its call. Both orders can be replayed.
- If a later request filter rejects a resumed step after HTTP 200 is
  committed, the stream can end without an Anthropic terminal event. If a
  model round is truncated or exceeds the connector's size limit after
  streaming starts, the connector emits one SSE `error` and stops.
- A toolset's `defer_loading: true` and pinned tool lists return HTTP 400.

See [`examples/configs/anthropic/mcp-connector.yaml`](../examples/configs/anthropic/mcp-connector.yaml).
Claude Code is unaffected: it runs MCP servers itself and sends their tools
as ordinary client tools, which every route serves.

### Committed example with credential isolation

[`examples/configs/anthropic/messages-native-vllm.yaml`](../examples/configs/anthropic/messages-native-vllm.yaml)
is the hardened version of the config above, used by a real Claude Code
client against native vLLM. It has no
`anthropic_messages_to_chat_completions[_stream]` filters, since translation
would be lossy, and isolates three credentials:

- `basic_auth` authenticates the client to the gateway. A trusted caller
  (for example Claude Code via `ANTHROPIC_CUSTOM_HEADERS`) sends an
  `Authorization: Basic ...` credential. With `strip_authorization: true` the
  filter removes it after verification, so it never reaches vLLM. The
  password comes from `GATEWAY_AUTH_PASSWORD` at pipeline build time.
- `headers` removes the client's `x-api-key`.
- `credential_injection` adds the backend's `Authorization: Bearer` token
  from `VLLM_API_KEY`, resolved at pipeline build time.

Only authenticated callers are served, and vLLM only sees the server-owned
bearer token. The `router` forwards `/v1/messages`,
`/v1/messages/count_tokens`, and `/` unchanged, so token counting and model
discovery stay native.

### Committed example that translates to Chat Completions

For a backend that speaks only Chat Completions,
[`examples/configs/anthropic/messages-to-openai-vllm.yaml`](../examples/configs/anthropic/messages-to-openai-vllm.yaml)
keeps the same three credential boundaries and adds the
`anthropic_messages_to_chat_completions[_stream]` filters plus a
`path_rewrite` from `POST /v1/messages` to `/v1/chat/completions`. The client
speaks Anthropic; vLLM only sees Chat Completions.

An unmodified Claude Code client always sends prompt caching
(`cache_control`) and extended thinking (`thinking`), which Chat Completions
cannot represent. It also sends a mid-conversation `system` message after its
first user turn, which some vLLM chat templates reject ("System message must
be at the beginning"; others render it in place). The example opts into
degradation so these requests succeed instead of returning 400:

```yaml
- filter: anthropic_messages_to_chat_completions
  allow_lossy_features:
    - prompt_caching
    - extended_thinking
    - mid_conversation_system
```

What each feature drops, and how the degradation is reported (`WARN` log,
`praxis_anthropic_messages_to_chat_completions_degraded_total`,
`x-degraded-features`), is on the
[filter page](filters/apis/anthropic_messages_to_chat_completions.md). Remove
an entry to make that feature fail closed again. An empty allowlist, the
default in [`messages-to-openai.yaml`](../examples/configs/anthropic/messages-to-openai.yaml),
keeps strict rejection.

`/v1/messages/count_tokens` has no Chat Completions equivalent, so the
`path_rewrite` is anchored to `^/v1/messages$` and leaves it alone. A
Chat-Completions-only backend returns 404, and Claude Code treats native
token counting as best-effort.

### Acceptance test: real Claude Code drives vLLM, native and transformed (issue #1025)

[`tests/integration/tests/suite/claude_code_vllm.rs`](../tests/integration/tests/suite/claude_code_vllm.rs)
runs a pinned Claude Code executable through a deterministic multi-step coding
task against real vLLM, once per config and permission mode, for four GPU
cases:

- `acceptEdits`, with the task's Read, Edit, and Bash tools preapproved: the
  deterministic baseline.
- `auto`, with no tools preapproved and `CLAUDE_CODE_AUTO_MODE_SERVER=0`:
  Claude Code must send classifier model requests through Praxis before it
  can Edit the file or run Bash.

```text
native      Claude Code ─► Praxis (messages-native-vllm.yaml)    ─► vLLM /v1/messages
transformed Claude Code ─► Praxis (messages-to-openai-vllm.yaml) ─► vLLM /v1/chat/completions
```

Each case asserts that the task finished: a successful exit without timeout,
the exact uppercase-derived output file, a verification marker written only
when the task's `verify.sh` confirms the compare, and a non-empty final
summary in the stream-json output. Wire fidelity and credential isolation are
tested deterministically against fake backends:

- Native passthrough (no `chat/completions` reshaping, exact served model on
  every inference body), credential isolation, and native token counting:
  [`tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs`](../tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs).
- Translation (`/v1/messages` rewritten to `/v1/chat/completions`, system
  hoisted into a Chat Completions message, the response translated back) and
  credential isolation:
  [`tests/integration/tests/suite/examples/anthropic_messages_to_openai_vllm.rs`](../tests/integration/tests/suite/examples/anthropic_messages_to_openai_vllm.rs).

In both, `x-api-key` and the gateway `Authorization: Basic` credential are
stripped, only the injected backend bearer reaches vLLM, and `basic_auth`
rejects an unauthenticated caller.

The tests skip unless every required variable is set. One vLLM container can
serve both surfaces:

```console
PRAXIS_TEST_CLAUDE_CODE_BIN=/absolute/path/to/claude \
PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
PRAXIS_TEST_VLLM_MODEL=<exact-served-model-name> \
VLLM_API_KEY=<backend-bearer-token> \
  cargo test -p praxis-tests-integration --test suite \
  claude_code_vllm::pinned_claude_code_drives -- --nocapture
```

Set `PRAXIS_TEST_CLAUDE_CODE_NETNS` (and `PRAXIS_TEST_LISTEN_ADDRESS` to the
host-side veth address, which Praxis then binds) to run the client in a Linux
network namespace that can reach only Praxis, which proves it cannot bypass
the proxy.

Praxis and vLLM do not implement Anthropic's server-side auto-mode
classifier protocol (`safeguards` and `safeguard_results`). Do not set
`CLAUDE_CODE_AUTO_MODE_SERVER=1` or rely on Claude Code's server-side
default with this backend. Set `CLAUDE_CODE_AUTO_MODE_SERVER=0` so Claude Code
sends its classifier requests through Praxis. Anthropic calls this the
"local" classifier path, but it still makes model requests; nothing runs on
the device. See
[Auto mode classifier billing](https://code.claude.com/docs/en/auto-mode-classifier-billing).

The Claude Code version and launch setup, vLLM image digest, served model,
and startup request matrix are pinned in
[`tests/integration/fixtures/claude-code-cli/pin.toml`](../tests/integration/fixtures/claude-code-cli/pin.toml).
The `vllm-gpu-claude-acceptance` job in
[`.github/workflows/vllm-integration.yaml`](../.github/workflows/vllm-integration.yaml)
runs the four scenarios sequentially against one vLLM container, each exactly
once (no retry), on every nightly GPU run. A fifth scenario on the same
container runs a read-only planning turn over the native path. It fails if
the run exhausts its turn budget or reasoning-channel text reaches the
answer, guarding against issue #1418, where a mismatched `--reasoning-parser`
made Claude Code loop and show its reasoning as the plan. The
`run_claude_acceptance` workflow-dispatch input runs the job on its own. The
job fails fast if any pin is empty or still contains `TBD`.

### Official Anthropic SDK against live vLLM

[`tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py`](../tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py)
runs the official Anthropic Python SDK against both committed vLLM examples:
non-streaming messages and usage, streamed event order and final text,
`messages.count_tokens`, and a `tool_use` → `tool_result` → final answer
loop. The translated config can count tokens here only because the shared
vLLM server also exposes the native endpoint. The live SDK step in
[`vllm-integration.yaml`](../.github/workflows/vllm-integration.yaml) runs
against the same Qwen3-8B instance as Claude Code acceptance. A named
tool-choice request returned `end_turn` on both routes in the first live run;
the test records that as an expected failure when a `tool_use` block is
present, and named tool choice remains unqualified.

Image coverage uses Qwen3-VL-4B-Instruct in the separate
[`anthropic-vllm-vision.yaml`](../.github/workflows/anthropic-vllm-vision.yaml)
GPU workflow, which sends generated red and blue PNGs through both routes and
checks that the model names each color. Both workflows set
`PRAXIS_TEST_REQUIRE_LIVE=1`, so missing infrastructure fails instead of
skipping. Label an upstream PR `vllm-full-suite` to run both before merge; the
vision workflow also runs on its own schedule and by manual dispatch once it
is on the default branch.

To run against a local keyed vLLM server:

```console
cargo build -p praxis-ai-proxy
PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
PRAXIS_TEST_VLLM_MODEL=<served-model-name> \
VLLM_API_KEY=<backend-bearer-token> \
PRAXIS_TEST_REQUIRE_LIVE=1 \
  uv run tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py -s -k "not image"
# For a separately served vision model, replace the final -k expression with: -k image
```

This qualifies those workflows and models only. It is not a Messages
conformance gate: other content blocks, request fields, server tools,
caching, and model-dependent behavior need their own coverage before a
broader support claim. The SDK is pinned to `anthropic==1.9.0` with its
`2023-06-01` API-version header as the qualified baseline; upgrading it is a
separate, validated change.

### Anthropic SDK v1/v2 compatibility lanes (issue #1825)

Anthropic Python SDK v2 changes the defaults the client puts on the wire:

| | SDK v1 (through 1.17.0) | SDK v2 |
|---|---|---|
| `anthropic-version` | `2023-06-01` | `2026-10-21` |
| credential header | `x-api-key: <key>` | `authorization: Bearer <key>` |

[`tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py`](../tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py)
runs both pinned lanes against local recording backends on native
`/v1/messages` passthrough and Messages -> Chat Completions translation. It
checks each lane's wire headers, JSON and streaming round-trips, tool use, and
the `2026-10-21` constraints each route owns (sampling parameter mapping,
empty versus active citations, `document` and `thinking` block rejection). It
needs no Anthropic credentials and makes no external calls.

The OpenAI -> Anthropic Messages direction uses the OpenAI SDK as the client,
so it does not vary by Anthropic SDK. Its backend headers (`anthropic-version`
from `api_version`, default `2023-06-01`; injected `x-api-key`; stripped
caller `authorization`) are asserted in
[`tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py`](../tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py).

Run both lanes with `make test-anthropic-sdk`, or one at a time:

```console
cargo build -p praxis-ai-proxy
uv run --with "anthropic==1.17.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -s -v --anthropic-sdk-version=1.x
uv run --with "anthropic==2.3.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -s -v --anthropic-sdk-version=2.x
```

CI runs both lanes in the `anthropic-sdk` job of
[`integration.yaml`](../.github/workflows/integration.yaml).

## API version compatibility

Version `2026-10-21` rejects requests that `2023-06-01` accepts:

- `temperature`, `top_p`, and `top_k`, even at default values
- `document` content blocks in assistant messages
- non-empty `citations` on user `text` blocks

SDK v2 sends `2026-10-21` by default, so upgrading a client SDK can change
the version the backend sees. Praxis does not reject these fields by version.
Each route either forwards the caller's or operator's version and lets the
backend validate, or ends the Anthropic contract.

| Route | `anthropic-version` sent to the backend | Caller credential | Sampling fields | Assistant `document`, user text citations |
|---|---|---|---|---|
| Native passthrough (`anthropic_messages_protocol`) | The caller's version; `default_version` only when the caller sends none | Forwarded unless the chain removes it | Forwarded; a `2026-10-21` backend rejects them | Forwarded; a `2026-10-21` backend rejects them |
| Messages to Chat Completions (`anthropic_messages_to_chat_completions`) | None; the header is removed | `x-api-key` and `authorization` removed | `temperature` and `top_p` mapped, `top_k` forwarded, under either version | Rejected with a 400 under either version: Chat Completions cannot represent them |
| Chat Completions to Messages (`openai_chat_completions_to_anthropic`) | The operator's `api_version` (default `2023-06-01`); caller headers ignored | `authorization` and `x-api-key` removed; `credential_inject` adds the provider key | `temperature` and `top_p` forwarded; a `2026-10-21` backend rejects them; `top_k` rejected by the bridge | Never produced: the translator emits only text, image, and tool blocks |

On native passthrough, `default_version` never overrides the caller, so an
SDK v1 client reaches the backend with `2023-06-01` and an SDK v2 client with
`2026-10-21` even when the chain sets `default_version: "2023-06-01"`. The
backend returns its own error for unsupported fields, and decides how to
treat a version it does not recognize.

For the Chat Completions to Messages translator, see the
[version and sampling contract](developing/openai-anthropic-compatibility.md#api-version-and-sampling).

### SDK credentials and gateway authentication

SDK v1 sends the key as `x-api-key`; SDK v2 sends `authorization: Bearer`
(see the [compatibility lanes](#anthropic-sdk-v1v2-compatibility-lanes-issue-1825)).
A chain that removes only `x-api-key` stops the v1 key but not the v2 key.
Pick the case that matches your deployment:

- If each client brings its own Anthropic key for the upstream, a pure
  passthrough forwards either header unchanged.
- If the gateway owns the upstream credential, remove both `x-api-key` and
  `authorization` with a `headers` `request_remove`, then add the upstream
  key with `credential_injection`. Pending removals are applied before
  injected values (`apply_pending_header_mutations`), so the order of those
  two filters doesn't matter. See
  [Passthrough to Anthropic API](#passthrough-to-anthropic-api).
- If the gateway authenticates the caller, as the committed vLLM examples
  do, run `basic_auth` with `strip_authorization: true`. It accepts only
  `Basic`, so
  an SDK v2 Bearer key gets a 401 before routing and never reaches the
  backend. To use SDK v2 with these gateways, send the gateway credential as
  the `Authorization` header; SDK v2 then sends it instead of its Bearer key:

  ```python
  import base64
  import os

  from anthropic import Anthropic

  token = base64.b64encode(f"gateway:{os.environ['GATEWAY_AUTH_PASSWORD']}".encode()).decode()
  client = Anthropic(
      base_url="http://127.0.0.1:8080",
      api_key="unused-by-gateway",
      default_headers={"Authorization": f"Basic {token}"},
  )
  ```

  SDK v1 sends that header next to `x-api-key`, which the examples' `headers`
  filter removes. Claude Code supplies the same header through
  `ANTHROPIC_CUSTOM_HEADERS`.

The Messages to Chat Completions translator removes both Anthropic credential
forms itself, since a Chat Completions backend can't tell an Anthropic key
from its own token.

## Passthrough to Anthropic API

Route to `api.anthropic.com`, injecting the gateway's key as `x-api-key`.
Here the gateway owns the upstream credential, so the chain also removes
`authorization`. `credential_injection` is configured for `x-api-key`, not
`authorization`, so without that removal an SDK v2 caller's Bearer key would
also reach Anthropic, and the request might authenticate as the caller. When
callers bring their own keys, see
[SDK credentials](#sdk-credentials-and-gateway-authentication). The `router`
must precede `credential_injection` so a cluster is selected.

```yaml
listeners:
  - name: gateway
    address: "127.0.0.1:8080"
    filter_chains: [anthropic]

filter_chains:
  - name: anthropic
    filters:
      - filter: anthropic_messages_request
        on_invalid: continue

      - filter: anthropic_messages_protocol
        default_version: "2023-06-01"

      - filter: headers
        request_set:
          - name: Host
            value: "api.anthropic.com"
        request_remove:
          - authorization

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

Translate Anthropic requests to Chat Completions for backends that speak only
OpenAI (for example llm-d with disaggregation, or KServe without Anthropic
support).

```yaml
listeners:
  - name: gateway
    address: "127.0.0.1:8080"
    filter_chains: [transform]

filter_chains:
  - name: transform
    filters:
      - filter: anthropic_messages_request
        on_invalid: continue

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

insecure_options:
  allow_private_endpoints: true # local development backend
```

`anthropic_messages_to_chat_completions` handles non-streaming bodies.
`anthropic_messages_to_chat_completions_stream`, gated on a
`text/event-stream` response, handles Chat Completions SSE, so ordinary JSON
responses stay on the buffered path.

Its credential removal, rejected fields, and lossy features are listed on the
[filter page](filters/apis/anthropic_messages_to_chat_completions.md). The
translation also:

- Hoists `system` to an OpenAI system message.
- Translates text, user image, assistant `tool_use`, and user `tool_result`
  blocks. Top-level `document`, `search_result`, and other blocks Chat
  Completions cannot represent are rejected.
- Translates `tool_result.is_error: true` by prefixing the tool message with
  `Anthropic tool_result error:` (a bare marker when the content is empty),
  since Chat Completions has no tool-result error flag.
- Splits array-form `tool_result.content`. Text parts join the tool message.
  Image parts move to a follow-up user message emitted after every tool
  message of the same Anthropic message, because Chat Completions requires
  all tool replies to follow the assistant `tool_calls` message directly.
  `search_result` and `document` parts are flattened to text. Unsupported part
  types, malformed image sources, and parts that flatten to nothing are
  rejected.
- Maps `stop_sequences` to `stop`, plus `tool_choice` semantics and tool
  definitions.
- Reports a matched stop sequence as `stop_reason: stop_sequence` with the
  matched value when the backend names it in vLLM's choice-level
  `stop_reason`. A backend that returns only `finish_reason: stop` (for
  example the OpenAI API) can't distinguish the two, so the response reports
  `end_turn`.
- Maps `metadata.user_id` to `safety_identifier` as its SHA-256 hex digest,
  `output_config.effort` to `reasoning_effort`, and a `json_schema`
  `output_config.format` (or the deprecated `output_format`) to a strict
  `response_format`. Any other `output_config` key gets a 400.
- Rejects the MCP connector (`mcp_servers` or any `mcp_toolset` tool) with a
  400 that names it, unless `anthropic_mcp_connector` resolved it earlier in
  the step (see [MCP connector](#mcp-connector)).
- Rejects with a 400 `service_tier`, `container`, `inference_geo`, and
  `mcp_servers`, whose effect the translated response cannot report, and the
  Chat Completions fields whose output it would discard (`n`, `logprobs`,
  `top_logprobs`, `audio`, `modalities`, `functions`, `function_call`,
  `web_search_options`, `moderation`). A `null` or the field's documented
  default (for example `n: 1`) is dropped instead.
- Rejects non-null `thinking` and `context_management`, and `thinking` blocks
  or blocks with non-empty citations or prompt-cache controls, unless the
  matching lossy feature is enabled.
- Forwards every other field (for example `top_k`) for the backend to
  validate.
- Keeps a text-only `system` message inside `messages` in place unless
  `mid_conversation_system` is enabled. With it, every `system` message after
  the first non-system turn folds into the single leading system message in
  encounter order, and leading `system` messages join the merged prompt in
  encounter order when a later message moves.
- Translates the response back to Anthropic format and keeps the original
  `finish_reason` in filter metadata as `openai.finish_reason`.
- Turns pre-stream upstream 4xx/5xx responses into Anthropic error envelopes
  for streaming and non-streaming requests, keeping the upstream status and
  request ID and clearing stale representation headers on rewritten bodies.

If an upstream success response cannot be translated, the filter rejects it,
which can return an HTTP 500 before response headers are sent. Once the
upstream success headers have been sent, the proxy aborts the body, and the
client may see a connection error or an incomplete HTTP 200 response.

## Running with Debug Logging

```console
RUST_LOG=debug cargo run -p praxis-ai-proxy -- -c config.yaml
```

Filter-specific logging:

```console
RUST_LOG=praxis_ai_apis=debug,praxis_ai_filters=debug cargo run -p praxis-ai-proxy -- -c config.yaml
```
