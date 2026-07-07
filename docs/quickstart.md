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

## Next steps

- [Example configs](../examples/README.md): working YAML
  for every feature.
- [Filters](filters/README.md): AI filters and how to
  write your own.
- [Praxis core](https://github.com/praxis-proxy/praxis):
  listener, filter-chain, routing, and load-balancer
  configuration.
