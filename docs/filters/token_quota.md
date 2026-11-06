# PPE token quota plugin

The experimental `token-quota-filter` feature registers the
`ai/token-quota` host plugin with the Praxis Policy Engine (PPE). It does not
register a `token_quota` HTTP filter or create a separate CEL policy surface.

`token_count` normalizes the provider's final usage and supplies it to the
`policy` filter as trusted response context. PPE selects an `llm:` route and
runs the configured quota plugin on `cmf.llm_output`. The plugin partitions
cache and reasoning tokens, applies its weights, rounds up, and returns a
typed result for `token_rate_limit` and `external_metering`.

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

Different routes can run different plugin instances, so PPE owns formula
selection by model or route. Backend/provider-specific overrides are not
currently available and are a follow-up gap. All weights must
be finite and non-negative. Omitted weights default to `1.0`; `unit` defaults
to `quota_units` and is not a currency.

The proxy chain must be declared in this request order:

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
```

Response hooks run in reverse, producing
`token_count -> policy/PPE -> consumers`. Metadata-only policy works for JSON
and streamed responses without buffering or exposing response content to the
quota plugin. The PPE result includes `value`, `formula_id`, `unit`, `status`,
and the exact normalized `usage` and routing `context` used for audit.
