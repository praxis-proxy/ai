# Screen requests and observe token usage

This recipe combines the existing [NeMo request-screening example](../../examples/configs/nemo-guardrails.yaml)
and [token-counting example](../../examples/configs/token-counting.yaml).
It screens supported Chat Completions text before inference, then extracts
OpenAI-format usage into metrics. It does not enable response screening.

## Prerequisites and build

Follow the [recipe prerequisites](README.md). Start an OpenAI-compatible Chat
backend at `127.0.0.1:3000`, and NeMo Guardrails **0.24.0+** at
`127.0.0.1:3001`. Deploy a NeMo configuration and replace `your-config` below
with its actual ID. The [guardrails reference](../filters/ai_guardrails.md)
describes the service contract and failure modes. Choose known passing and
blocking prompts from your deployed rails; a generic prompt is not guaranteed
to trigger a particular policy.

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

The backend credential belongs only to cluster `backend`; it is not attached
to NeMo calls. This loopback recipe assumes no NeMo authentication. For a
protected NeMo service use the [service-account example](../../examples/configs/nemo-guardrails-service-account.yaml)
with HTTPS and the correct trust roots. Gateway-owned inference secrets also
require a client authentication gate before exposing this listener externally.

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

For a passing prompt, expect Chat Completions JSON and provider usage.
After completion, `/metrics` should contain `praxis_ai_prompt_tokens_total`
and `praxis_ai_output_tokens_total` when usage is supplied and parsed. A
blocked request returns 403 and must not reach inference; verify backend
request counts. A changed Chat prompt can be forwarded with masked text;
non-Chat redaction has narrower support. NeMo errors must not be interpreted
as a successful policy check. The default failure mode is closed.

For streaming, add `"stream":true` to the JSON and `-N` to curl.
`stream_usage_inject` requests final SSE usage from supporting Chat backends.
Counts become available when that usage arrives; missing usage or a truncated
stream cannot provide complete billing evidence. Request screening still
runs before forwarding. Do not enable response screening for this streaming
recipe: Chat response-phase screening rejects streaming requests with 422
until incremental output evaluation is supported.

## Limits, identity, and accounting

`token_count` reads JSON/SSE usage from the configured provider dialect. Choose
the dialect it actually observes when composing translators. Provider-reported
cache and reasoning fields may be missing, and translation can lose details;
review the [compatibility matrix](../conformance/translation-matrix.md).
Body-derived counts arrive after response headers: `token_usage_headers`
does not turn these metrics into `Praxis-Token-*` headers merely by reordering
filters. The Bedrock InvokeModel header-only path is a separate case.

For bounded per-tenant counters, declare [token_usage_metrics](../filters/token_usage_metrics.md)
before `token_count`, provide explicit tenant/model allowlists, and derive
identity from authenticated metadata. Client headers are insufficient.
Prometheus labels do not provide authorization or billing-grade settlement.
[External metering](../../examples/configs/external-metering-file-key.yaml)
adds balance checks and usage reporting to a separate service.

Usage extraction does not enforce token limits. Experimental
[token_rate_limit](../filters/token_rate_limit.md) reserves estimated budgets
and reconciles them; [token_ceiling](../filters/token_ceiling.md) constrains
the final provider-bound body. Both require their own features and policies.

## Cleanup

Stop the proxy with Ctrl+C, then stop services you started solely for the
recipe. Unset `BACKEND_API_KEY` in the proxy shell and `MODEL` in the client
shell; remove `guardrails-usage.yaml` if no longer needed. Keep the admin
metrics endpoint private in a deployed service.
