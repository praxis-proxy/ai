# Quickstart

This source-build guide is for operators and client integrators who want to
forward a first request to a hosted OpenAI API. Praxis AI is alpha software. The
default responder check needs no provider account; the API request does. For
other protocols and providers, start with [Client and provider
paths](client-provider-paths.md).

## Prerequisites and build

You need Git, Make, curl, and the native build requirements listed in
[Development setup](developing/getting-started.md#requirements). Run the
commands from a repository checkout, as a non-root user:

```console
git clone https://github.com/praxis-proxy/ai.git
cd ai
```

If you already have a checkout, use its root directory. This guide follows that
checkout's code and feature set; use the documentation matching your binary.

Build the release binary:

```console
make release
```

## Check the default responder

Start Praxis AI:

```console
./target/release/praxis-ai
```

With `PRAXIS_CONFIG` unset and no `praxis.yaml` in the current directory, the
server uses its built-in default configuration and listens on `127.0.0.1:8080`.
Verify it:

```console
curl http://127.0.0.1:8080/
```

```json
{"status": "ok", "server": "praxis-ai"}
```

## Route to an AI backend

Stop the default responder with Ctrl+C before continuing. The configured proxy
below uses the same port, but replaces the built-in responder configuration.

For the hosted request, you need outbound DNS and HTTPS access to
`api.openai.com`, trusted CA certificates, an OpenAI API account with API
access, and access to the model you request. Obtain an API key from your
provider account, then set it in the shell where you will run curl:

```console
export OPENAI_API_KEY='<your-openai-api-key>'
export OPENAI_MODEL='<model-available-to-your-account>'
```

Replace both placeholders. The commands below do not print the key, but the
request sends it to the configured upstream. This example forwards the caller's
Authorization header; it does not configure gateway-owned credentials. For
routing-selected upstream secrets, see [credential
injection](filters/credential_inject.md) and its routing prerequisites.

Create `praxis-ai.yaml`:

```yaml
listeners:
  - name: ai
    address: "127.0.0.1:8080"
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

Validate the configuration, then start Praxis AI:

```console
./target/release/praxis-ai --validate -c praxis-ai.yaml
./target/release/praxis-ai -c praxis-ai.yaml
```

Leave the proxy running. In the shell with your key and model variables set,
send a request through its `/v1` route:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{\"model\": \"$OPENAI_MODEL\", \"input\": \"Hello\"}"
```

A successful request returns an upstream Responses JSON object with generated
output. Model availability, billing, and response content are controlled by the
provider. The initial `/` check only proves the default listener responds; this
request additionally depends on the upstream service.

This configuration forwards native Responses traffic using ordinary routing. It
does not install the managed Responses filters, local Conversations storage, or
an agentic loop. `make release` selects the standard proxy feature set; managed
Responses requires `openai-responses`, and the ephemeral agent loop additionally
requires `openai-agentic-loop`. These are experimental opt-ins; see [Cargo
features](features.md#cargo-features) before choosing a build.

## Connect an existing client

For an OpenAI client that accepts a base URL, set it to
`http://127.0.0.1:8080/v1` and keep the provider API key and an accessible
model. The `/v1` suffix is part of the base URL. For example, with the OpenAI
Python package installed, run this in the shell where you exported the
variables:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)
response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input="Hello",
)
print(response.output_text)
```

This calls native `/v1/responses`. The route also forwards
`/v1/chat/completions` without translating it; use the API supported by your
upstream. Other provider protocols need their own configuration, as listed in
[Client and provider paths](client-provider-paths.md).

## Chat Completions

Use the same prerequisites, standard release binary, credentials, full
`praxis-ai.yaml`, validation, and startup commands above. This path forwards
Chat Completions natively; no Responses feature is needed. With the proxy
running, send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{\"model\":\"$OPENAI_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

Expect an upstream Chat Completions object with `choices[0].message.content`
and provider-reported `usage`. A tool-using response can instead ask the client
to run a function; this configuration does not execute it. For Python, replace
the Responses call above with:

```python
completion = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[{"role": "user", "content": "Hello"}],
)
print(completion.choices[0].message.content)
```

For SSE, add `"stream":true` to the curl JSON and `-N` to curl. Expect
`data:` chunks and the provider's terminal `[DONE]`; use a streaming SDK parser
rather than parsing the whole body as one JSON object. This ordinary forwarding
chain does not translate events, aggregate usage, or screen output. For a local
compatible backend, change the endpoint, Host/TLS settings, and credentials to
match that backend; the hosted configuration above targets OpenAI specifically.
Use the failure checks and cleanup below for either API.

## If the request fails

- **Connection refused on port 8080:** check that Praxis AI is running and its
  startup log has no configuration error. Stop the default responder before
  starting the configured proxy.
- **Unknown filter type at startup:** check the binary's [build
  features](features.md#cargo-features) against the configured filters.
- **HTTP error from the API:** inspect the response body. Authentication,
  permissions, model access, quota, and request validation can fail
  independently of proxy routing. Check the key and model in the client shell
  without printing the key.
- **No upstream response:** inspect the proxy log and check DNS, outbound HTTPS,
  CA trust, and the route's cluster, Host header, and TLS SNI. Do not disable
  TLS verification to work around a configuration error.

The SDK and hosted request above require provider access. For credential-free,
local fixture coverage, see the deterministic OpenAI Agents SDK test command in
[Development
setup](developing/getting-started.md#openai-agents-sdk-responses-compatibility).
That fixture exercises a managed Responses configuration with a different build;
it does not verify this hosted route or live provider compatibility.

## Cleanup

Stop the configured proxy with Ctrl+C. In the client shell, remove the
variables:

```console
unset OPENAI_API_KEY OPENAI_MODEL
```

Remove `praxis-ai.yaml` if you no longer need the example.

## Next steps

- [Example configs](../examples/README.md): working YAML for supported features
  and integration patterns.
- [Filters](filters/README.md): operator capabilities and composition.
- [Praxis core](https://github.com/praxis-proxy/praxis): listener, filter-chain,
  routing, and load-balancer configuration.
