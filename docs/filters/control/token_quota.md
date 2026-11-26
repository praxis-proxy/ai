# PPE token quota plugin

The experimental `token-quota-filter` feature registers the
`ai/token-quota` host plugin with the Praxis Policy Engine (PPE). It does not
register a `token_quota` HTTP filter or create a separate CEL policy surface.

`token_count` normalizes the provider's final usage and passes it to the
`policy` filter as trusted response context. PPE selects an `llm:` route and
runs the configured quota plugin on `cmf.llm_output`. The plugin separates
cache and reasoning tokens, applies its weights, rounds up, and returns a
typed result for `token_rate_limit` and `external_metering`.

`token_count` reads `openai`, `anthropic`, `google`, `bedrock`,
`bedrock_invoke_model`, and `azure` response usage. On the request side,
`policy` must see and admit a usable top-level `model` in the JSON request
body; a model only in the URL does not count. Missing models fail closed by
default through the policy filter's `llm.require_model` behavior.

Configure the PPE plugin in the policy document:

```yaml
plugins:
  - name: weighted-quota-v1
    kind: ai/token-quota
    hooks: [cmf.llm_output]
    capabilities: [metadata_only_response]
    mode: sequential
    on_error: fail
    config:
      formula_id: weighted-quota-v1
      unit: quota_units
      weights:
        input: 0.3333333333333333
        cached_input: 0.03333333333333333
        cache_write: 0.4166666666666667
        output: 1.0
        reasoning: 1.0

routes:
  - llm: "*"
    authorization:
      post_invocation:
        - "run(weighted-quota-v1)"
```

Routes can run different plugin instances, so PPE chooses the formula by
model or route. Backend- or provider-specific overrides are not available yet.
Weights must be finite and non-negative; omitted weights default to `1.0`.
`unit` defaults to `quota_units` and is not a currency.

Declare the proxy chain in this request order:

```yaml
- filter: token_rate_limit       # optional consumer
  use_token_quota: true
  # ...
- filter: external_metering      # optional consumer
  # ...
- filter: policy
  config_path: token-quota-policy.yaml
  body_access: read_only
- filter: token_count
  provider: openai
  max_body_bytes: 1048576       # optional; default JSON capture limit (1 MiB)
```

Response hooks run in reverse: `token_count -> policy/PPE -> consumers`.
Metadata-only policy works for JSON and streamed responses without showing
response content to the quota plugin. `token_count` processes SSE
incrementally but buffers non-streaming JSON up to `max_body_bytes` (1 MiB by
default). A larger JSON response produces `usage_overflow`, and the limiter
keeps its reservation estimate instead of charging actual usage.

OpenAI Chat Completions streaming requests must set
`stream_options: {"include_usage": true}` (or use `stream_usage_inject`) so the
final stream event includes usage. Otherwise the quota result is
`missing_usage` and the limiter keeps its reservation estimate.

Shared quota requires the policy's default `body_access: read_only`;
`read_write` response buffering does not yet publish the quota result to the
limiter. A metadata-only denial is recorded in PPE output for the host
consumer but cannot stop a response whose content was already released. For
audit, the PPE result includes `value`, `formula_id`, `unit`, `status`, and
the exact normalized `usage` and routing `context` it used.
