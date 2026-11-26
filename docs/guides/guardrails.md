# Screen requests with guardrails

Screen Chat Completions text with NeMo Guardrails before inference and record
OpenAI-format usage as metrics. The configuration combines the
[NeMo request-screening example](../../examples/configs/nemo-guardrails.yaml)
and the [token-counting example](../../examples/configs/token-counting.yaml).
Response screening stays off. For reading the usage metrics see
[Observe and meter token usage](token-usage.md); for budgets see
[Limit token usage](token-limits.md).

## Prerequisites and build

Follow the [guide prerequisites](README.md). Start an OpenAI-compatible Chat
backend at `127.0.0.1:3000` and NeMo Guardrails 0.24.0+ at `127.0.0.1:3001`.
Deploy a NeMo configuration and put its ID in place of `your-config` below. The
[guardrails reference](../filters/control/ai_guardrails.md) covers the service
contract and failure modes. Pick test prompts your deployed rails are known to
pass and block; a generic prompt may not trigger any policy.

```console
cargo build -p praxis-ai-proxy
export BACKEND_API_KEY='<backend-key>'
```

Save this complete local configuration as `guardrails-usage.yaml`:

```yaml
admin:
  metrics_address: "127.0.0.1:9901"
listeners:
  - name: gateway
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ai_operation
      - filter: model_to_header
      - filter: ai_guardrails
        outbound_chain: nemo-outbound
        provider:
          type: nemo
          endpoint: "http://127.0.0.1:3001/v1/checks"
          guardrails:
            config_ids: ["your-config"]
          timeout_ms: 5000
          max_message_checks: 32
        phase:
          request: true
          response: false
      - filter: stream_usage_inject
      - filter: token_count
        provider: openai
      - filter: router
        routes:
          - path_prefix: "/v1/chat/completions"
            cluster: backend
      - filter: credential_injection
        clusters:
          - name: backend
            header: Authorization
            env_var: BACKEND_API_KEY
            header_prefix: "Bearer "
            strip_client_credential: true
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:3000"]
  - name: nemo-outbound
    filters:
      - filter: request_id
insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
```

The backend credential goes only to cluster `backend`, never to NeMo calls.
This loopback recipe assumes NeMo needs no authentication; for a protected NeMo
service use the [service-account example](../../examples/configs/nemo-guardrails-service-account.yaml)
with HTTPS and the correct trust roots. Add client authentication before
exposing this listener, since it holds a gateway-owned inference secret.

## Validate, run, and inspect

```console
./target/debug/praxis-ai --validate -c guardrails-usage.yaml
./target/debug/praxis-ai -c guardrails-usage.yaml
```

In a client shell, replace the model placeholder:

```console
export MODEL='<served-model>'
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
curl --fail-with-body http://127.0.0.1:9901/metrics
```

A passing prompt returns Chat Completions JSON with provider usage, and the
token counters appear in `/metrics` as described in
[token usage](token-usage.md#verify). A blocked request returns 403 and never
reaches inference; confirm with backend request counts. When NeMo modifies a
Chat prompt, the masked text can be forwarded; redaction outside Chat has
narrower support. A NeMo error is not a passed check: the default failure mode
is closed.

For streaming, add `"stream":true` to the JSON and `-N` to curl. Request
screening still runs before forwarding. Keep response screening off here:
with `phase.response` enabled, streaming Chat requests are rejected with 422
until incremental output evaluation is supported.

## Cleanup

Stop the proxy with Ctrl+C and any services you started for this recipe.
Unset `BACKEND_API_KEY` in the proxy shell and `MODEL` in the client shell, and
remove `guardrails-usage.yaml`.

## Filter reference

- [`ai_operation`](../filters/core/ai_operation.md)
- [`model_to_header`](../filters/routing/model_to_header.md)
- [`ai_guardrails`](../filters/control/ai_guardrails.md)
- [`stream_usage_inject`](../filters/observability/stream_usage_inject.md)
- [`token_count`](../filters/observability/token_count.md)
