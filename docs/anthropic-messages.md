# Anthropic Messages API

Praxis AI can forward Anthropic Messages (`/v1/messages`) to a native
backend, or translate supported requests to a Chat Completions backend.
Start with the [complete operator recipes](recipes/anthropic.md) for build,
credentials, startup, requests, and cleanup. This page owns the detailed
protocol and qualification notes; the [client/provider table](client-provider-paths.md)
also covers the optional Vertex Anthropic path.

Translation is a bounded compatibility layer. Review rejected fields, explicit
lossy opt-ins, and [known gaps](conformance/translation-matrix.md#known-gaps)
before choosing it over a native backend.

## Filters

| Filter | Purpose |
| ------ | ------- |
| `anthropic_messages_request` | Process the create-message body once: validate the envelope and promote routing facts |
| `anthropic_messages_protocol` | Header management for native `/v1/messages` backends |
| `anthropic_messages_to_chat_completions` | Bidirectional body transformation to the Chat Completions wire shape |
| `anthropic_messages_to_chat_completions_stream` | Incremental Chat Completions SSE to Messages events; see compatibility limits below |

## Passthrough to vLLM

Route Anthropic requests directly to a backend that
supports `/v1/messages` natively (e.g. vLLM with
Anthropic endpoint enabled).

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

Chat Completions cannot represent the Anthropic-only features an unmodified
Claude Code client always sends — prompt caching (`cache_control` markers) and
extended thinking (`thinking`). The example enables **operator-approved
degradation** on the translation filter so those requests succeed instead of
failing closed with a 400:

```yaml
- filter: anthropic_messages_to_chat_completions
  allow_lossy_features:
    - prompt_caching
    - extended_thinking
```

Each listed feature's wire markers are validated and then stripped, and the
degradation is reported to the operator — one `WARN` log per request, a
`praxis_anthropic_messages_to_chat_completions_degraded_total` counter per
feature, and an `x-degraded-features` response header. `prompt_caching`
preserves all prompt and tool content but does not honor explicit cache
breakpoints, so cost and latency may differ; `extended_thinking` means the
translated response carries no thinking blocks. Remove an entry to make that
feature fail closed again. Malformed markers, other `context_management` edits,
and every feature not listed are still rejected with a 400. Leaving the
allowlist empty (the default of [`messages-to-openai.yaml`](../examples/configs/anthropic/messages-to-openai.yaml))
keeps the strict reject behavior, which an operator who would rather see the
unsupported request fail than silently lose the feature should prefer.

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
exactly once (no retry), on every nightly GPU run. A fifth scenario on the same
container drives a read-only planning turn over the native path and fails if the
run exhausts its turn budget instead of answering, or if reasoning-channel text
reaches the user-visible answer — the nightly regression guard for issue #1418,
where a mismatched `--reasoning-parser` made Claude Code loop and render its
reasoning as the plan. It can also run independently
through the `run_claude_acceptance` workflow-dispatch input. Runtime pins must be
complete; the job fails fast with an explanatory error if any required value is
empty or still contains `TBD`.

### Official Anthropic SDK against live vLLM

[`tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py`](../tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py)
uses the official Anthropic Python SDK against both committed vLLM examples.
The text matrix checks non-streaming messages and usage, streamed event order
and final text, `messages.count_tokens`, and automatic `tool_use` followed by a
`tool_result` and final answer. The translated config can count tokens in this
matrix because the shared vLLM server also exposes the native count-tokens
endpoint; a Chat-Completions-only backend cannot. The live SDK step in
[`vllm-integration.yaml`](../.github/workflows/vllm-integration.yaml) runs this
matrix against the same Qwen3-8B instance as Claude Code acceptance.
The first live run returned `end_turn` for a named tool-choice request on both
routes. The SDK test records that stop-reason mismatch as an expected failure
when a `tool_use` block is present; named tool-choice semantics remain
unqualified.

Positive image coverage uses Qwen3-VL-4B-Instruct in the separate
[`anthropic-vllm-vision.yaml`](../.github/workflows/anthropic-vllm-vision.yaml)
GPU workflow. It sends generated red and blue PNGs through the SDK and both
Praxis routes, then checks that the vision model identifies each color. The
Qwen3-8B text model cannot establish image support. Both workflows set
`PRAXIS_TEST_REQUIRE_LIVE=1`, so missing infrastructure fails rather than
silently skipping the tests. Label an upstream PR `vllm-full-suite` to run both
GPU workflows before merge; the vision workflow also runs on its own schedule
and after a manual dispatch once it exists on the default branch.

To run either matrix against a local keyed vLLM server, build Praxis and set
the exact served model name and backend key:

```console
cargo build -p praxis-ai-proxy
PRAXIS_TEST_VLLM_BASE_URL=http://127.0.0.1:8000 \
PRAXIS_TEST_VLLM_MODEL=<served-model-name> \
VLLM_API_KEY=<backend-bearer-token> \
PRAXIS_TEST_REQUIRE_LIVE=1 \
  uv run tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py -s -k "not image"
# For a separately served vision model, replace the final -k expression with: -k image
```

This matrix qualifies those named workflows and models. It is not an
Anthropic Messages conformance gate: other content blocks, request fields,
server tools, caching, and model-dependent behaviors require separate contract
and live-backend coverage before a broader support claim.

The GPU acceptance SDK is pinned to `anthropic==1.9.0` deliberately: that
client release and its `2023-06-01` API-version header are the qualified
baseline, and upgrading it is a separate, explicitly validated change rather
than something a floating dependency may drift into.

### Anthropic SDK v1/v2 compatibility lanes (issue #1825)

Anthropic Python SDK v2 changes the defaults the client puts on the wire:

| | SDK v1 (through 1.17.0) | SDK v2 |
|---|---|---|
| `anthropic-version` | `2023-06-01` | `2026-10-21` |
| credential header | `x-api-key: <key>` | `authorization: Bearer <key>` |

[`tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py`](../tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py)
exercises both pinned lanes against local recording backends across the two
routes where the Anthropic SDK is the client — native `/v1/messages`
passthrough and the Messages -> Chat Completions translation — asserting each
lane's wire headers, JSON and streaming message round-trips, tool use, and
the route-owned `2026-10-21` acceptance constraints (sampling parameter
mapping, empty versus active citations, `document` and `thinking` block
rejection). It needs no Anthropic credentials and makes no external calls.
On the translated route, the `anthropic_messages_to_chat_completions`
filter strips the caller's Anthropic wire credentials — `x-api-key` (SDK v1)
and `authorization: Bearer` (SDK v2) — along with `anthropic-version`, so
neither reaches a Chat Completions backend in either lane.

The OpenAI -> Anthropic Messages direction is driven by the OpenAI SDK as the
client, so it does not vary on the Anthropic SDK axis; its backend wire
headers (operator-selected `anthropic-version` via `api_version`,
defaulting to `2023-06-01`, injected `x-api-key`,
stripped caller `authorization`) are asserted in
[`tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py`](../tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py).

Run both lanes locally with `make test-anthropic-sdk`, or one lane at a time:

```console
cargo build -p praxis-ai-proxy
uv run --with "anthropic==1.17.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -s -v --anthropic-sdk-version=1.x
uv run --with "anthropic==2.3.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -s -v --anthropic-sdk-version=2.x
```

CI runs both lanes as sequential steps in the `anthropic-sdk` job of
[`integration.yaml`](../.github/workflows/integration.yaml).

## API version compatibility

Anthropic Messages API version `2026-10-21` rejects requests that
`2023-06-01` accepts:

- `temperature`, `top_p`, and `top_k`, even when they carry default values
- `document` content blocks in assistant messages
- non-empty `citations` on user `text` blocks

Anthropic SDK v2 selects `2026-10-21` by default, so upgrading a client SDK
can change which version reaches the backend. Praxis does not reject these
fields based on version. Each route either forwards the version chosen by the
caller or the operator and lets the backend validate it, or ends the Anthropic
contract entirely.

| Route | `anthropic-version` sent to the backend | Caller credential | Sampling fields | Assistant `document`, user text citations |
|---|---|---|---|---|
| Native passthrough (`anthropic_messages_protocol`) | The caller's version; `default_version` only when the caller sends none | Forwarded unless the chain removes it | Forwarded; a `2026-10-21` backend rejects them | Forwarded; a `2026-10-21` backend rejects them |
| Messages to Chat Completions (`anthropic_messages_to_chat_completions`) | None; the header is removed | `x-api-key` and `authorization` removed | `temperature` and `top_p` mapped, `top_k` forwarded, under either version | Rejected with a 400 under either version: Chat Completions cannot represent them |
| Chat Completions to Messages (`openai_chat_completions_to_anthropic`) | The operator's `api_version` (default `2023-06-01`); caller headers ignored | `authorization` and `x-api-key` removed; `credential_inject` adds the provider key | `temperature` and `top_p` forwarded; a `2026-10-21` backend rejects them; `top_k` rejected by the bridge | Never produced: the translator emits only text, image, and tool blocks |

Native passthrough preserves a caller-supplied `anthropic-version` and the
request body. `anthropic_messages_protocol.default_version` fills in a
missing version; it does not override the caller or enforce version-specific
body restrictions. An SDK v1 client therefore reaches the backend with
`2023-06-01` and an SDK v2 client with `2026-10-21`, even when the chain sets
`default_version: "2023-06-01"`. A backend that accepts the newer version
returns its own error for unsupported fields; how a backend treats a version
it does not recognize is up to that backend.

For Chat Completions clients translated to Anthropic Messages, the operator
selects `openai_chat_completions_to_anthropic.api_version`. Its default is
`2023-06-01`. Under either supported version, supported sampling values reach
the backend unchanged; the backend owns version-specific validation and any
rejection is returned through the existing error translation. The translator
setting wins over a co-configured native protocol default in either
filter order. It applies to the whole translator configuration, not individual
routing-overlay providers. Compatible Messages-to-Chat Completions and native vLLM
routes retain their sampling support. See the
[version and sampling contract](developing/openai-anthropic-compatibility.md#api-version-and-sampling).

### SDK credentials and gateway authentication

SDK v1 sends the API key as `x-api-key`; SDK v2 sends it as
`authorization: Bearer` (see the
[compatibility lanes](#anthropic-sdk-v1v2-compatibility-lanes-issue-1825)).
A chain that removes only `x-api-key` stops the SDK v1 key, but not the
SDK v2 key. Pick the case that matches your deployment:

- **The caller's key is meant for the upstream.** A pure passthrough
  forwards either header unchanged. This is correct when every client brings
  its own Anthropic key.
- **The gateway owns the upstream credential.** Remove both `x-api-key` and
  `authorization` with a `headers` `request_remove`, then add the upstream
  key with `credential_injection`. Pending removals run before injected
  values, so the filter order doesn't matter. See
  [Passthrough to Anthropic API](#passthrough-to-anthropic-api).
- **The gateway authenticates the caller.** The committed vLLM examples run
  `basic_auth` with `strip_authorization: true`. It accepts only `Basic`, so
  an SDK v2 Bearer key is rejected with a 401 before routing and never
  reaches the backend. To use SDK v2 with these gateways, send the gateway
  credential as the `Authorization` header. SDK v2 then sends it in place of
  its Bearer key rather than alongside it:

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

  SDK v1 sends that header next to `x-api-key`. The examples' `headers`
  filter removes `x-api-key`. Claude Code supplies the same header through
  `ANTHROPIC_CUSTOM_HEADERS`.

The Messages to Chat Completions translator removes both Anthropic
credential forms on its own: a Chat Completions backend can't tell an
Anthropic key from its own token.

## Passthrough to Anthropic API

Route to `api.anthropic.com` with credential
injection for the `x-api-key` header.

A pure passthrough forwards whichever credential header the caller sent, so an
SDK v2 client's `Authorization: Bearer` key reaches Anthropic unchanged. That is
intended when the caller supplies its own key. When the gateway owns the
upstream credential instead, add `authorization` to a `headers` `request_remove`
in the same chain. `credential_injection` is configured for `x-api-key`, not
`authorization`, so a client Bearer would otherwise reach Anthropic too, and the
request may then authenticate as the caller rather than as the gateway. The
relative order of header removal and injection does not matter: pending removals
are applied before injected values in `apply_pending_header_mutations`. The
`router` must still precede `credential_injection` so a cluster is selected.

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

Transform Anthropic requests to OpenAI Chat
Completions format for backends that only speak
OpenAI (e.g. llm-d with disaggregation, KServe
without Anthropic support).

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

The `anthropic_messages_to_chat_completions` filter:
- Removes both client credential forms (the legacy `x-api-key` header and the
  `Authorization: Bearer` header SDK v2 sends) because a Chat Completions
  backend cannot tell an Anthropic key from its own. Pending header removals
  are applied before injected values, so a `credential_injection` in the same
  chain still installs the backend's own token
- Hoists `system` to an OpenAI system message
- Translates text, user image, assistant `tool_use`,
  and text-only user `tool_result` blocks; rejects
  `document`, `search_result`, and other blocks that
  Chat Completions cannot represent
- Rejects `tool_result.is_error: true` because Chat
  Completions has no equivalent tool-result error flag
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
- Rejects non-null `thinking` and `context_management`
  because Chat Completions has no equivalent, unless
  `extended_thinking` is in `allow_lossy_features`, which
  strips `thinking` and thinking-only `context_management`
  edits and reports the degradation instead (other
  `context_management` edits are still rejected)
- Forwards every other field untouched (for example
  `top_k`) and leaves its validation to the backend
- Rejects `thinking` content blocks and content blocks
  carrying non-empty citations or prompt-cache controls,
  unless `prompt_caching` is in `allow_lossy_features`,
  which strips `cache_control` markers (preserving the
  prompt and tool content) and reports the degradation
- Transforms the response back to Anthropic format
- Normalizes pre-stream upstream 4xx/5xx responses into
  Anthropic error envelopes for both streaming and
  non-streaming requests
- Preserves the upstream status and request ID while
  clearing stale representation headers on rewritten
  response bodies
- Preserves original `finish_reason` in filter
  metadata as `openai.finish_reason`

If an upstream success response cannot be translated, the filter rejects it.
This can return an HTTP 500 before response headers are sent. Once the
upstream success headers have been sent, the proxy aborts the response body;
the client may observe a connection error or an incomplete HTTP 200 response.

Add `anthropic_messages_to_chat_completions_stream` with a `text/event-stream`
response condition when the backend may return streaming
Chat Completions SSE. Keep it response-gated so normal
JSON responses stay on the buffered `anthropic_messages_to_chat_completions`
path.

## Filter Configuration Reference

### `anthropic_messages_request`

Owns the Anthropic create-message body. The Messages operation registry decides
from the request head whether this filter applies, so `POST /v1/messages` is
processed and every other operation — token counting, the batch family — and
every other path is released untouched.

```yaml
filter: anthropic_messages_request
on_invalid: continue      # continue | reject
max_body_bytes: 1048576    # 1 MiB
headers:
  format: x-praxis-ai-format
  model: x-praxis-ai-model
  stream: x-praxis-ai-stream
```

The matched body is deserialized once. That parse produces the routing
metadata, the promoted headers, the filter results, and the typed
`AnthropicMessagesState`. The forwarded bytes are never rewritten here.

Body classification precedence no longer applies to this surface. The filter
this replaced ranked `input`, then `messages` plus `max_tokens` with Anthropic
structural signals, then `messages` alone — so an ordinary Anthropic body could
be read as Chat Completions until a `/v1/messages` or `anthropic-version`
tiebreak corrected it. The registry removes that guesswork.

The endpoint is the authority on protocol. A Chat Completions-shaped body on
`/v1/messages` is still the create-message operation, and an Anthropic-shaped
body on another path is not. `anthropic-version` is contract validation rather
than operation identity, so its absence changes nothing.

This replaced a pair of filters that inferred the protocol from body structure
and broke ties on `/v1/messages` or `anthropic-version`. That precedence is
gone: the body no longer votes on which protocol a request belongs to.

`on_invalid` governs the envelope — a missing, malformed, or non-object body:

| Value | Behavior |
| ----- | -------- |
| `continue` (default) | Forward the body and let the backend answer |
| `reject` | Refuse at the gateway with an Anthropic-compatible client error |

Those were previously separate filters with contradictory answers, and which
one applied depended on whether a chain included the validator. Choose
explicitly.

### `anthropic_messages_protocol`

Injects `anthropic-version` header if absent.
A caller-supplied version, including the SDK v2 default `2026-10-21`, is
forwarded unchanged. No body transformation, version-specific validation,
or credential handling. See [API version compatibility](#api-version-compatibility).

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
