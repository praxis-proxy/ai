# Quickstart

Build the release binary:

```console
make release
```

Start Praxis AI:

```console
./target/release/praxis-ai
```

The server starts on `127.0.0.1:8080` with a built-in
default configuration. Verify it:

```console
curl http://127.0.0.1:8080/
```

```json
{"status": "ok", "server": "praxis-ai"}
```

## Route to an AI backend

Create `praxis-ai.yaml`:

```yaml
listeners:
  - name: ai
    address: "127.0.0.1:8080"
    filter_chains: [openai]

filter_chains:
  - name: openai
    filters:
      - filter: openai_responses_format
      - filter: router
        routes:
          - path_prefix: "/v1"
            cluster: openai_backend
      - filter: load_balancer
        clusters:
          - name: openai_backend
            endpoints:
              - "api.openai.com:443"
            tls:
              sni: "api.openai.com"
```

Start Praxis AI with your config:

```console
./target/release/praxis-ai -c praxis-ai.yaml
```

Requests to port 8080 are now forwarded to the OpenAI
API:

```console
curl http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-4o", "input": "Hello"}'
```

## Deployment

The same YAML serves ingress, egress, or in-cluster
placements; change listener address, TLS, and upstream
clusters per environment. Filter chains stay the same.
See [AI Gateway overview](overview.md) for use-cases.

## Next steps

- [AI Gateway overview](overview.md)
- [OpenAI Responses](openai-responses.md) and
  [Anthropic Messages](anthropic-messages.md) guides
- [Example configs](../examples/README.md)
- [Filters](filters/README.md)
- [Praxis core](https://github.com/praxis-proxy/praxis)
