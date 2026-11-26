# Run an Anthropic Messages client

Use a native Messages backend whenever it serves the operations you need, and
translation only for a Chat-Completions-only backend. These local recipes reuse
the credential-isolated examples. Protocol behavior, SDK version differences,
hosted Anthropic configuration, and qualification evidence are in the
[Anthropic guide](../anthropic-messages.md).

## Prerequisites and build

Follow the [guide prerequisites](README.md). Start a keyed backend on
`127.0.0.1:8000` with a model you have qualified. Native mode needs `/v1/messages`;
translated mode needs `/v1/chat/completions`. For model serving and GPU setup,
including pins and parser requirements, see
[the vLLM guide](../developing/cli-vllm-through-praxis.md). Not every vLLM
release or model supports Messages.

Both examples use experimental Basic Auth to keep caller and backend
credentials apart. Build and set variables in the proxy shell:

```console
cargo build -p praxis-ai-proxy --features basic-auth-filter
export GATEWAY_AUTH_PASSWORD='<gateway-password>'
export VLLM_API_KEY='<backend-key>'
```

In a second, client shell, set the same gateway password and the exact
served model:

```console
export GATEWAY_AUTH_PASSWORD='<gateway-password>'
export MODEL='<exact-served-model-name>'
```

These examples listen on loopback, use plain HTTP to a loopback backend, and
allow private endpoints. For remote access, add TLS and an authenticated
ingress. Do not expose the catch-all route as a public API allowlist.

## Native Messages

The [native configuration](../../examples/configs/anthropic/messages-native-vllm.yaml)
authenticates the client, processes the Messages envelope, supplies a missing
version, strips client keys, injects the backend bearer, and forwards bodies
and SSE untranslated.

In the proxy shell:

```console
./target/debug/praxis-ai --validate -c examples/configs/anthropic/messages-native-vllm.yaml
./target/debug/praxis-ai -c examples/configs/anthropic/messages-native-vllm.yaml
```

In the client shell:

```console
curl --fail-with-body -u "gateway:$GATEWAY_AUTH_PASSWORD" \
  http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":128,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}]}"
```

Expect a Messages object with `type: message`, text in `content`, a
`stop_reason`, and `usage`. Add `"stream":true` and `curl -N` for
`message_start`, content events, and `message_stop`. The client handles any
`tool_use` it receives. Token counting works only if the backend implements
`/v1/messages/count_tokens`.

For the hosted Anthropic API, save the
[hosted configuration](../anthropic-messages.md#passthrough-to-anthropic-api)
as `anthropic-hosted.yaml`, export `ANTHROPIC_API_KEY='<provider-key>'` in the
proxy shell, and run the same `--validate -c` and startup commands with that
file. Send the curl above without `-u`, with `MODEL` set to an Anthropic model
you can access. That config sets Host and TLS/SNI for Anthropic and removes
caller authorization before injecting `x-api-key`, but has no gateway
authentication: keep it in a trusted local environment or put an ingress gate
in front before sharing gateway-owned credentials.

## Translated Messages

Stop the native proxy and start the
[translation configuration](../../examples/configs/anthropic/messages-to-openai-vllm.yaml)
with the same build and environment:

```console
./target/debug/praxis-ai --validate -c examples/configs/anthropic/messages-to-openai-vllm.yaml
./target/debug/praxis-ai -c examples/configs/anthropic/messages-to-openai-vllm.yaml
```

Repeat the Messages curl. The backend now receives `POST /v1/chat/completions`
and the client still gets an Anthropic Messages object. Streaming goes
through the included response-gated SSE adapter. Gateway Basic Auth and
caller Anthropic keys are removed; only `VLLM_API_KEY` reaches the backend.

For CLI compatibility the example opts into dropping prompt-cache markers and
extended thinking controls, and reports each degradation with
`x-degraded-features`, a warning, and a counter. For strict behavior, copy the
YAML locally and remove `allow_lossy_features`; those controls then return
400. Document blocks and server tools remain unrepresentable; an
error-marked tool result reaches the backend as a tool message prefixed
`Anthropic tool_result error:`. The
[count-tokens endpoint has no Chat equivalent](../anthropic-messages.md#transformation-to-openai-backend),
so a Chat-only backend returns 404 on that unchanged path.

See the [translation matrix](../conformance/translation-matrix.md#known-gaps)
for stream termination/error, reasoning, and usage gaps. A successful text
request does not show full Anthropic conformance or qualify named tool
choice. Native passthrough preserves a caller's version; translation has a
narrower versioned contract (see the [Anthropic guide](../anthropic-messages.md#api-version-compatibility)).

## Chat client to selected Messages provider

For the reverse direction, reuse the
[selected-provider configuration](../../examples/configs/openai-selected-provider-legacy-translation.yaml).
The adapter acts only on the trusted profile from `intelligent_route`; a model
name or client header cannot enable it. See the
[compatibility contract](../developing/openai-anthropic-compatibility.md).

You need a trusted, versioned [routing overlay](../filters/routing/intelligent_route.md#credentials-and-provider-gateways)
with a `messages` provider profile, the selected model available on the
backend, and the referenced API key mounted under
`<mount-base>/<namespace>/<secret-name>/<key>`. The overlay must contain the
stable ID, target model, path, and `apikey` reference the selection contract
expects. Its content digest does not authenticate the publisher, so the
operator must authorize both the overlay publisher and the mounted secrets.

```console
cargo build -p praxis-ai-proxy
cp examples/configs/openai-selected-provider-legacy-translation.yaml selected-messages.yaml
```

In the copy, bind the listener to `127.0.0.1`, set `overlay_file` to your
trusted overlay's absolute path, and set `projected_credential_mount_base` to
your protected mount root. Keep `model_to_header`, `intelligent_route`, the SSE
adapter, the finite adapter, and `credential_inject` in their existing order.
Keep `api_translation_enabled: true` only on an operator-authorized route.
`api_version` applies to the whole translator instance, not to individual
overlay entries.

The committed endpoints are local test backends on ports 8001/8002. Either
start native Messages/Chat backends there with the models the overlay names,
or replace the cluster endpoints with the real provider addresses and
matching TLS SNI. For hosted Anthropic, use `api.anthropic.com:443` with
matching SNI, set the Host header to `api.anthropic.com` only on that
provider's path, and remove the private-endpoint exceptions if all endpoints
are public. For a first test, use an overlay with only the Anthropic
candidate, an Anthropic-only load-balancer cluster, and `openai-backend`
removed from `intelligent_route.clusters`, so no Host override is shared
across providers. Add gateway authentication before exposing this
server-owned key remotely.

```console
./target/debug/praxis-ai --validate -c selected-messages.yaml
./target/debug/praxis-ai -c selected-messages.yaml
```

In the client shell, choose the public model name from the overlay:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<overlay-public-model>","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions JSON even though the backend receives Messages. Add
`"stream":true` and `-N` for translated Chat SSE. Missing selected secrets
fail closed, and a different caller bearer does not change the trusted
provider profile. The translator ignores `n`, `seed`, `parallel_tool_calls`,
and `user` after type checks, rejects other unsupported fields, and defaults
an omitted token limit to 4096; read the compatibility contract before
relying on scheduling, reproducibility, or multiple completions.

Afterwards, stop the proxy and any test backends you started, delete
`selected-messages.yaml` and any disposable key copies you made, and leave
shared projected mounts and trusted overlay sources alone.

## Failure checks and cleanup

A missing or wrong gateway password is rejected before inference. A backend
401/403 points at the backend credential or permissions. A 400 from
translation often means a rejected field rather than a model failure; read
the dialect-shaped error. For connection failures, check port 8000, backend
logs, the exact model name, and the proxy startup log.

Stop the proxy with Ctrl+C, then stop the backend only if you started it for
this recipe. In both shells as applicable:

```console
unset GATEWAY_AUTH_PASSWORD VLLM_API_KEY MODEL ANTHROPIC_API_KEY
```

Delete only locally created copies such as `anthropic-hosted.yaml`.

## Filter reference

- [`ai_operation`](../filters/core/ai_operation.md)
- [`anthropic_messages_protocol`](../filters/apis/anthropic_messages_protocol.md)
- [`anthropic_messages_request`](../filters/apis/anthropic_messages_request.md)
- [`anthropic_messages_to_chat_completions`](../filters/apis/anthropic_messages_to_chat_completions.md)
- [`anthropic_messages_to_chat_completions_stream`](../filters/apis/anthropic_messages_to_chat_completions_stream.md)
- [`model_to_header`](../filters/routing/model_to_header.md)
- [`intelligent_route`](../filters/routing/intelligent_route.md)
- [`credential_inject`](../filters/providers/credential_inject.md)
- [`openai_chat_completions_to_anthropic`](../filters/apis/openai_chat_completions_to_anthropic.md)
- [`openai_chat_completions_to_anthropic_stream`](../filters/apis/openai_chat_completions_to_anthropic_stream.md)

Core filters such as `router`, `headers`, `credential_injection`, and
`load_balancer` are documented in the [Praxis core filter reference](https://github.com/praxis-proxy/praxis/blob/main/docs/filters/reference.md).
