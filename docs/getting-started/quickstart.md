# Quickstart

Run the published Praxis AI container image and forward a first OpenAI
Responses request through it. Praxis AI is alpha software. To build the binary
yourself instead, follow [Build from source](build-from-source.md); for other
protocols and providers, start with
[Choose a client and provider](choose-client-and-provider.md).

## Prerequisites

- Docker or Podman. The commands below use `docker`; Podman accepts the same
  arguments.
- An image tag matching the documentation you are reading. Released images are
  published as `ghcr.io/praxis-proxy/ai:<version>` (for example `0.12.1`); see
  [image tags](../developing/release.md#image-tags). The published image is
  built for `linux/amd64`; on other hosts, configure AMD64 emulation in your
  container runtime and keep `--platform linux/amd64`.
- `curl`, outbound DNS and HTTPS access from the container to
  `api.openai.com`, an OpenAI API key, and access to the model you request.

The published image is built with the default `full` feature set. Experimental
managed Responses, agentic loops, and stores are not included; see
[container details](../developing/release.md#container-details) and
[Cargo features](../features.md#cargo-features).

## Configure the gateway

Save this as `praxis-ai.yaml` in the current directory:

```yaml
listeners:
  - name: ai
    address: "0.0.0.0:8080"
    filter_chains: [openai]

filter_chains:
  - name: openai
    filters:
      - filter: ai_operation
      - filter: router
        routes:
          - path_prefix: "/v1"
            cluster: openai_backend
      - filter: headers
        request_set:
          - name: Host
            value: api.openai.com
      - filter: load_balancer
        clusters:
          - name: openai_backend
            endpoints:
              - "api.openai.com:443"
            tls:
              sni: "api.openai.com"
```

This is the [build-from-source configuration](build-from-source.md#route-to-an-ai-backend)
with one change: the listener binds `0.0.0.0` so the container port can be
published. The host side stays on loopback in the commands below.

## Validate and start

The image entrypoint is `praxis-ai` and its working directory is
`/etc/praxis`. Mount the configuration read-only and pass the path with `-c`.
Set the version you selected:

```console
export PRAXIS_AI_VERSION='<version>'
```

Validate the configuration:

```console
docker run --rm --platform linux/amd64 \
  --volume "$PWD/praxis-ai.yaml:/etc/praxis/praxis-ai.yaml:ro" \
  "ghcr.io/praxis-proxy/ai:$PRAXIS_AI_VERSION" \
  --validate -c /etc/praxis/praxis-ai.yaml
```

Successful validation exits with status `0` without starting the gateway.
Then start it, publishing the proxy port only on host loopback:

```console
docker run --rm --name praxis-ai --platform linux/amd64 \
  --publish 127.0.0.1:8080:8080 \
  --volume "$PWD/praxis-ai.yaml:/etc/praxis/praxis-ai.yaml:ro" \
  "ghcr.io/praxis-proxy/ai:$PRAXIS_AI_VERSION" \
  -c /etc/praxis/praxis-ai.yaml
```

Leave it running and use another terminal for the request.

## Send a request

Set your key and a model available to your account in the client shell:

```console
export OPENAI_API_KEY='<your-openai-api-key>'
export OPENAI_MODEL='<model-available-to-your-account>'
```

```console
curl --fail-with-body http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{\"model\": \"$OPENAI_MODEL\", \"input\": \"Hello\"}"
```

This example forwards the caller's Authorization header to OpenAI; it does not
configure gateway-owned credentials.

## Expected result

A successful request returns HTTP `200` and an upstream Responses JSON object
with generated output. Model availability, billing, and response content are
controlled by the provider. An HTTP error with an OpenAI error body (for
example `401` for an invalid key) still shows that the gateway reached the
provider; check the key, model access, and quota. For connection and startup
failures, see [If the request fails](build-from-source.md#if-the-request-fails).

This pipeline classifies and forwards native Responses traffic. It does not
enable gateway-managed conversation storage, tool execution, guardrails, or
client authentication. The same route also forwards `/v1/chat/completions`
unchanged; see [Chat Completions](build-from-source.md#chat-completions) and
[Connect an existing client](build-from-source.md#connect-an-existing-client).

## Cleanup

Stop the container with Ctrl+C (the `--rm` flag removes it). In the client
shell, remove the variables:

```console
unset OPENAI_API_KEY OPENAI_MODEL PRAXIS_AI_VERSION
```

Remove `praxis-ai.yaml` if you no longer need it.

## Next steps

- [Choose a client and provider](choose-client-and-provider.md): protocol,
  credentials, feature selection, and compatibility limits.
- [Guides](../guides/README.md): task guides for Anthropic, routing, providers,
  guardrails, token usage, and managed Responses.
- [Example configs](../../examples/README.md): working YAML for supported
  features and integration patterns.
