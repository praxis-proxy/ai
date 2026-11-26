# Build from source

Build Praxis AI from a checkout and forward a first request to the hosted
OpenAI API. To run the published container image instead, follow the
[Quickstart](quickstart.md). Praxis AI is alpha software. The default
responder check needs no provider account; the API request does. For other
protocols and providers, see
[Choose a client and provider](choose-client-and-provider.md).

## Prerequisites and build

You need Git, Make, curl, and the native build requirements in
[Development setup](../developing/getting-started.md#requirements). Work from a
checkout as a non-root user:

```console
git clone https://github.com/praxis-proxy/ai.git
cd ai
```

An existing checkout works too; this guide follows that checkout's code and
features. Build the release binary:

```console
make release
```

## Check the default responder

Start Praxis AI:

```console
./target/release/praxis-ai
```

With `PRAXIS_CONFIG` unset and no `praxis.yaml` in the current directory, the
server uses its built-in default configuration on `127.0.0.1:8080`. Check it:

```console
curl http://127.0.0.1:8080/
```

```json
{"status": "ok", "server": "praxis-ai"}
```

## Route to an AI backend

Stop the default responder with Ctrl+C. The proxy below uses the same port.

The hosted request needs outbound DNS and HTTPS to `api.openai.com`, trusted CA
certificates, an OpenAI API account, and access to the model you request. In
the shell where you will run curl, set:

```console
export OPENAI_API_KEY='<your-openai-api-key>'
export OPENAI_MODEL='<model-available-to-your-account>'
```

The commands below do not print the key, but the request sends it upstream.
This example forwards the caller's Authorization header and holds no
gateway-owned credentials. For upstream secrets chosen by routing, see
[credential injection](../filters/providers/credential_inject.md) and its
routing prerequisites.

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

Leave the proxy running and send a request through its `/v1` route:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{\"model\": \"$OPENAI_MODEL\", \"input\": \"Hello\"}"
```

A successful request returns a Responses JSON object from OpenAI.

This configuration forwards native Responses traffic through ordinary routing.
It does not install the managed Responses filters, local Conversations storage,
or an agentic loop. `make release` builds the standard feature set; managed
Responses needs `openai-responses`, and the ephemeral agent loop also needs
`openai-agentic-loop`. Both are experimental; see
[Cargo features](../features.md#cargo-features).

## Connect an existing client

Point an OpenAI client's base URL at `http://127.0.0.1:8080/v1` (the `/v1` is
part of the base URL) and keep the provider key and model. With the OpenAI
Python package installed, in the shell where you exported the variables:

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
`/v1/chat/completions` untranslated. Other provider protocols need their own
configuration; see [Choose a client and provider](choose-client-and-provider.md).

## Chat Completions

Chat Completions uses the same binary, credentials, `praxis-ai.yaml`, and
startup as above, and needs no Responses feature. With the proxy running:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{\"model\":\"$OPENAI_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

Expect a Chat Completions object with `choices[0].message.content` and
provider-reported `usage`. A tool-using response may ask the client to run a
function; this configuration does not run it. In Python, replace the Responses
call above with:

```python
completion = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[{"role": "user", "content": "Hello"}],
)
print(completion.choices[0].message.content)
```

For SSE, add `"stream":true` to the curl JSON and `-N` to curl. Expect `data:`
chunks ending with the provider's `[DONE]`, and use a streaming SDK parser
instead of parsing the body as one JSON object. This chain does not translate
events, aggregate usage, or screen output. For a local compatible backend,
change the endpoint, Host/TLS settings, and credentials; the configuration
above targets OpenAI only.

## If the request fails

- **Connection refused on port 8080:** check that Praxis AI is running and its
  startup log shows no configuration error. Stop the default responder before
  starting the configured proxy.
- **Unknown filter type at startup:** compare the binary's
  [build features](../features.md#cargo-features) with the configured filters.
- **HTTP error from the API:** read the response body. Authentication,
  permissions, model access, quota, and request validation fail independently
  of proxy routing. Check the key and model without printing the key.
- **No upstream response:** read the proxy log and check DNS, outbound HTTPS,
  CA trust, and the route's cluster, Host header, and TLS SNI. Do not disable
  TLS verification to get around a configuration error.

For credential-free local coverage, see the deterministic OpenAI Agents SDK
test in [Development setup](../developing/getting-started.md#openai-agents-sdk-responses-compatibility).
It runs a managed Responses configuration with a different build, so it does
not verify this hosted route or live provider compatibility.

## Cleanup

Stop the proxy with Ctrl+C and clear the variables:

```console
unset OPENAI_API_KEY OPENAI_MODEL
```

Delete `praxis-ai.yaml` if you no longer need it.

## Next steps

- [Example configs](../../examples/README.md)
- [Filters](../filters/README.md): capabilities and composition
- [Praxis core](https://github.com/praxis-proxy/praxis): listener, filter-chain,
  routing, and load-balancer configuration
