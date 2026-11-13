# Run an Anthropic Messages client

Use a native Messages backend whenever it serves the needed operations. Choose
translation only for a Chat-Completions-only backend. These local recipes reuse
the credential-isolated examples; the detailed [Anthropic guide](../anthropic-messages.md)
continues to own protocol behavior, SDK version differences, hosted Anthropic
configuration, and qualification evidence.

## Prerequisites and build

Follow the [recipe prerequisites](README.md). Start a keyed backend on
`127.0.0.1:8000` with a model you have qualified. Native mode needs `/v1/messages`;
translated mode needs `/v1/chat/completions`. Model serving and GPU setup follow
[the existing vLLM guide](../developing/cli-vllm-through-praxis.md), including
its pins and parser requirements. These recipes do not provision a GPU or
claim every vLLM release/model supports Messages.

Both committed examples use experimental Basic Auth to isolate caller and
backend credentials. Build and set variables in the proxy shell:

```console
cargo build -p praxis-ai-proxy --features basic-auth-filter
export GATEWAY_AUTH_PASSWORD='<gateway-password>'
export VLLM_API_KEY='<backend-key>'
```

Use the same gateway password in a second, client shell, and choose the exact
served model there:

```console
export GATEWAY_AUTH_PASSWORD='<gateway-password>'
export MODEL='<exact-served-model-name>'
```

These examples listen on loopback, speak HTTP to a loopback backend, and opt
into private endpoints. Use TLS and an authenticated ingress for remote access.
Do not expose the catch-all route as a public API allowlist.

## Native Messages

The [complete native configuration](../../examples/configs/anthropic/messages-native-vllm.yaml)
authenticates the client, classifies the operation, processes the Messages
envelope, supplies a missing version, strips client keys, routes, injects the
backend bearer, and forwards without translating bodies or SSE.

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

Expect a native Messages object with `type: message`, text in `content`, a
`stop_reason`, and `usage`. Add `"stream":true` and `curl -N` to receive native
`message_start`, content events, and `message_stop`. Function tools returned
as `tool_use` remain the client's responsibility. Token counting works only
if the backend implements `/v1/messages/count_tokens`.

For the hosted Anthropic API, use the full
[hosted configuration](../anthropic-messages.md#passthrough-to-anthropic-api),
export `ANTHROPIC_API_KEY='<provider-key>'` in the proxy shell, save that YAML
as `anthropic-hosted.yaml`, and run the same `--validate -c` and startup commands
with that file. Use the curl above without `-u`, replacing `MODEL` with an
accessible Anthropic model. That hosted sample has no gateway authentication:
restrict it to a trusted local environment or add an ingress gate before
sharing gateway-owned credentials. It sets Host and TLS/SNI for Anthropic and
removes caller authorization before injecting `x-api-key`.

## Translated Messages

Stop the native proxy. Use the
[complete isolated translation configuration](../../examples/configs/anthropic/messages-to-openai-vllm.yaml)
with the same build and environment:

```console
./target/debug/praxis-ai --validate -c examples/configs/anthropic/messages-to-openai-vllm.yaml
./target/debug/praxis-ai -c examples/configs/anthropic/messages-to-openai-vllm.yaml
```

Repeat the Messages curl above. The backend now receives
`POST /v1/chat/completions`, while the client still receives an Anthropic
Messages object. Streaming requires the included response-gated SSE adapter.
Gateway Basic Auth and caller Anthropic keys are removed; only `VLLM_API_KEY`
reaches the backend.

This example explicitly opts into dropping prompt-cache markers and extended
thinking controls for CLI compatibility. It reports actual degradation with
`x-degraded-features`, a warning, and a degradation counter. To require strict
behavior, copy the YAML to a local file and remove `allow_lossy_features`;
those unsupported controls then return 400. Document blocks, server tools,
and error-marked tool results remain unrepresentable. The
[count-tokens endpoint has no Chat equivalent](../anthropic-messages.md#transformation-to-openai-backend).
A Chat-only backend returns 404 on that unchanged path.

Review the [translation matrix](../conformance/translation-matrix.md#known-gaps)
for stream termination/error, reasoning, and usage gaps. A successful text
request is not evidence of full Anthropic conformance or named tool-choice
qualification. Native passthrough preserves a caller's version; translation
has a narrower versioned contract described in the existing guide.

## Chat client to selected Messages provider

For the reverse translation direction, reuse the
[complete selected-provider configuration](../../examples/configs/openai-selected-provider-legacy-translation.yaml).
The adapter consumes the trusted profile from `intelligent_route`; a model name
or an arbitrary client header cannot enable it. See the
[compatibility contract](../developing/openai-anthropic-compatibility.md).

Prerequisites are a trusted, versioned [routing overlay](../filters/intelligent_route.md#credentials-and-provider-gateways)
with a `messages` provider profile, the selected model available on the backend,
and the referenced API key mounted under
`<mount-base>/<namespace>/<secret-name>/<key>`. The overlay must contain the
stable ID, target model, path, and `apikey` reference expected by the selection
contract. Its content digest does not authenticate the publisher. The operator
must authorize both the overlay publisher and the mounted secrets.

```console
cargo build -p praxis-ai-proxy
cp examples/configs/openai-selected-provider-legacy-translation.yaml selected-messages.yaml
```

In the copy, bind the listener to `127.0.0.1`, set `overlay_file` to your
trusted overlay's absolute path, and set `projected_credential_mount_base` to
your protected mount root. Keep `model_to_header`, `intelligent_route`, the SSE
adapter, the finite adapter, and `credential_inject` in the existing order.
Keep `api_translation_enabled: true` only for an operator-authorized route.
The configured `api_version` applies to that translator instance, not separate
overlay entries.

The committed endpoints are local test backends on ports 8001/8002. Either
start native Messages/Chat backends there with the models named by the overlay,
or replace their cluster endpoints with the actual provider addresses and
matching TLS SNI. For hosted Anthropic, use `api.anthropic.com:443` and matching
SNI, set the Host header to `api.anthropic.com` only on that provider's path,
and remove local private-endpoint exceptions if all endpoints are public.
For a first test, use an overlay containing only the Anthropic candidate and
an Anthropic-only load-balancer cluster, also removing `openai-backend` from
`intelligent_route.clusters`; this avoids a shared Host override
across different providers. Add a gateway authentication boundary before
exposing this server-owned key remotely.

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

Expect Chat Completions JSON even though the selected backend receives
Messages. Add `"stream":true` and `-N` for translated Chat SSE. Missing
selected secrets fail closed; changing a caller bearer does not change the
trusted provider profile. This translator ignores some typed fields (`n`,
`seed`, `parallel_tool_calls`, `user`) and rejects other unsupported fields;
its default omitted token limit is 4096. Review the compatibility contract
before relying on scheduling, reproducibility, or multiple completions.

Stop the proxy and test backends you started, remove `selected-messages.yaml`,
and remove only disposable test key copies you created. Keep shared projected
mounts and trusted overlay sources intact.

## Failure checks and cleanup

A missing/wrong gateway password must be rejected before inference. Backend
401/403 indicates a separate backend credential or permission issue. A 400
from translation can indicate a rejected field rather than model failure;
inspect its dialect-shaped error. For connection failures, check port 8000,
backend logs, the exact model name, and the proxy startup log.

Stop the proxy with Ctrl+C, then stop the backend only if you started it for
this recipe. In both shells as applicable:

```console
unset GATEWAY_AUTH_PASSWORD VLLM_API_KEY MODEL ANTHROPIC_API_KEY
```

Delete only locally created copies such as `anthropic-hosted.yaml`.
