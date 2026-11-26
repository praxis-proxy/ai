# Quickstart

Run the published Praxis AI container image and forward an OpenAI Responses
request through it. Praxis AI is alpha software. To build the binary yourself,
follow [Build from source](build-from-source.md). For other protocols and
providers, see [Choose a client and provider](choose-client-and-provider.md).

## Prerequisites

- Docker or Podman. The commands use `docker`; Podman takes the same arguments.
- An image tag that matches the documentation you are reading. Released images
  are `ghcr.io/praxis-proxy/ai:<version>` (for example `0.12.1`); see
  [image tags](../developing/release.md#image-tags). The image is built for
  `linux/amd64`. On other hosts, configure AMD64 emulation in your container
  runtime and keep `--platform linux/amd64`.
- `curl`, outbound DNS and HTTPS from the container to `api.openai.com`, an
  OpenAI API key, and access to the model you request.

The image uses the default `full` feature set, which leaves out experimental
managed Responses, agentic loops, and stores. See
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
except that the listener binds `0.0.0.0` so the container port can be
published. The host side stays on loopback.

## Validate and start

The image entrypoint is `praxis-ai` and its working directory is
`/etc/praxis`. Mount the configuration read-only, pass its path with `-c`, and
set the version you picked:

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

Validation exits with status `0` and does not start the gateway. Then start
it, publishing the proxy port on host loopback only:

```console
docker run --rm --name praxis-ai --platform linux/amd64 \
  --publish 127.0.0.1:8080:8080 \
  --volume "$PWD/praxis-ai.yaml:/etc/praxis/praxis-ai.yaml:ro" \
  "ghcr.io/praxis-proxy/ai:$PRAXIS_AI_VERSION" \
  -c /etc/praxis/praxis-ai.yaml
```

Leave it running and open another terminal.

## Send a request

Set your key and a model your account can use:

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

The gateway forwards the caller's Authorization header to OpenAI. It holds no
credentials of its own here.

## Expected result

A successful request returns HTTP `200` and a Responses JSON object from
OpenAI. An HTTP error with an OpenAI error body (for example `401` for a bad
key) still means the gateway reached the provider; check the key, model
access, and quota. For connection and startup failures, see
[If the request fails](build-from-source.md#if-the-request-fails).

This pipeline classifies and forwards native Responses traffic. It adds no
conversation storage, tool execution, guardrails, or client authentication.
The same route forwards `/v1/chat/completions` unchanged; see
[Chat Completions](build-from-source.md#chat-completions) and
[Connect an existing client](build-from-source.md#connect-an-existing-client).

## Cleanup

Stop the container with Ctrl+C (`--rm` removes it), then clear the variables:

```console
unset OPENAI_API_KEY OPENAI_MODEL PRAXIS_AI_VERSION
```

Delete `praxis-ai.yaml` if you no longer need it.

## Next steps

- [Choose a client and provider](choose-client-and-provider.md)
- [Guides](../guides/README.md) for Anthropic, routing, providers, guardrails,
  token usage, and managed Responses
- [Example configs](../../examples/README.md)
